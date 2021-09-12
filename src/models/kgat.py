import logging
from dataclasses import dataclass
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch as th
import torch.optim as optim
from utils.helper import (ensure_file, generate_path_att_file,
                          generate_path_model_file, is_best_epoch)
from utils.loaders.kgat import KgatDataset
from utils.metrics import compute_metrics
from utils.sampler import generate_cf_batch, generate_kg_batch
from utils.types import KgatTrainType, UserItems

from .base import BasePredictor
from .kgat_torch import KGAT


@dataclass
class KgatPredictor(BasePredictor):
    # parameters
    model: KGAT
    epochs: int
    dim_entity_embed: int
    dim_relation_embed: int
    batch_size: int
    n_loop_cf: int
    n_loop_kg: int
    list_reg_learning_rates: List[float]  # for [loss_cf, loss_kge]
    learning_rate: float
    list_dim_conv_layer: List[int]
    list_size_dropout: List[float]
    top_ks: List[int]
    interval_evaluate: int
    stopping_steps: int

    # save
    best_epoch: int
    best_model: KGAT
    best_attentions: List[float]

    # dataset
    dataset: KgatDataset

    def __init__(
        self,
        epochs=500,
        dim_entity_embed=64,
        dim_relation_embed=64,
        batch_size=1024,
        list_reg_learning_rates=[1e-5, 1e-5],
        learning_rate=1e-4,
        list_dim_conv_layer=[64, 32, 16],
        list_size_dropout=[0.1, 0.1, 0.1],
        top_ks=[20, 60, 100],
        interval_evaluate=10,
        stopping_steps=300,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.dim_entity_embed = dim_entity_embed
        self.dim_relation_embed = dim_relation_embed
        self.batch_size = batch_size
        self.list_reg_learning_rates = list_reg_learning_rates
        self.learning_rate = learning_rate
        self.list_dim_conv_layer = list_dim_conv_layer
        self.list_size_dropout = list_size_dropout
        self.top_ks = top_ks
        self.interval_evaluate = interval_evaluate
        self.stopping_steps = stopping_steps
        self.best_epoch = 0

    def load(self, dataset: KgatDataset) -> None:
        self.model = KGAT(
            dataset=dataset,
            dim_entity_embed=self.dim_entity_embed,
            dim_relation_embed=self.dim_relation_embed,
            list_reg_learning_rates=self.list_reg_learning_rates,
            list_dim_conv_layer=self.list_dim_conv_layer,
            list_size_dropout=self.list_size_dropout,
            learning_rate=self.learning_rate,
        )
        self.dataset = dataset
        self.graph_train = dataset.graph_train
        self.graph_test = dataset.graph_test
        self.n_loop_cf = dataset.n_train // self.batch_size + 1
        self.n_loop_kg = dataset.n_triplet_kg // self.batch_size + 1

        logging.debug('Success to load torch KGAT model.')

    def train(self) -> None:
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.to(device)
        # trans_w (nn.Parameter) is not shown in log but it will be trained.
        # see also:
        # https://github.com/pytorch/pytorch/issues/4761
        # https://discuss.pytorch.org/t/unable-to-get-nn-parameter-as-a-parameter-in-model-parameters/115431
        logging.info(self.model)

        # consider: Adam? AdamW?
        # same as tensorflow: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate, eps=1e-7)

        # ids
        userids_batch_test = [th.LongTensor(self.dataset.uniq_userids[i:i + 10000]).to(
            device) for i in range(0, self.dataset.n_users, 10000)]
        itemids_kg = th.arange(self.dataset.n_items, dtype=th.long).to(
            device) + self.dataset.n_users

        # apply graph to device
        self.graph_train = self.graph_train.to(device)
        self.graph_test = self.graph_test.to(device)

        # train model
        time_start_train = time()
        self.model = self.model.train()
        list_hit, list_recall, list_precision, list_ndcg = [], [], [], []

        # 0. initialize attention scores
        with th.no_grad():
            self.graph_train.edata['attention'] = self.model.initialize_attention(
                self.graph_train)

        for epoch in range(1, self.epochs + 1):
            time_start_epoch = time()

            # 1. optimize recommendation (CF) part
            loss_cf, loss_reg = 0.0, 0.0
            for _ in range(self.n_loop_cf):
                users, pos_items, neg_items = generate_cf_batch(
                    self.dataset.dict_train_pos, self.batch_size, self.dataset.n_users, self.dataset.n_items)
                cf_batch_user = th.LongTensor(users).to(device)
                cf_batch_pos_item = th.LongTensor(pos_items).to(device)
                cf_batch_neg_item = th.LongTensor(neg_items).to(device)

                loss_cf_batch, loss_reg_batch = self.model(
                    KgatTrainType.COMP_LOSS_CF, self.graph_train, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
                loss_batch = loss_cf_batch + loss_reg_batch
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                loss_cf += loss_cf_batch.item()
                loss_reg += loss_reg_batch.item()

            # 2. update attention scores
            with th.no_grad():
                att = self.model.compute_attention(self.graph_train)
            self.graph_train.edata['attention'] = att

            # 3. optimize recommendation (KG) part
            loss_kg = 0.0
            for _ in range(self.n_loop_kg):
                heads, relations, pos_tails, neg_tails, epsilons = generate_kg_batch(
                    self.dataset.dict_h_tre_all, self.batch_size, self.dataset.n_entities_ckg_all)
                kg_batch_head = th.LongTensor(heads).to(device)
                kg_batch_relation = th.LongTensor(relations).to(device)
                kg_batch_pos_tail = th.LongTensor(pos_tails).to(device)
                kg_batch_neg_tail = th.LongTensor(neg_tails).to(device)
                kg_batch_epsilon = th.FloatTensor(epsilons).to(device)

                loss_kg_batch, loss_reg_batch = self.model(
                    KgatTrainType.COMP_LOSS_KG, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_epsilon)
                loss_batch = loss_kg_batch + loss_reg_batch
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                loss_kg += loss_kg_batch.item()
                loss_reg += loss_reg_batch.item()

            logging.info(
                '[Training] Epoch {:04d} / {:04d} [{:.1f}s] : CF(base) Loss {:.4f} : KG(kge) Loss {:.4f} : Regularization(reg) Loss {:.4f}'.format(
                    epoch, self.epochs, time() - time_start_epoch, loss_cf, loss_kg, loss_reg))

            # 4. evaluation
            if (epoch % self.interval_evaluate) == 0:
                time_start_eval = time()
                hits, precisions, recalls, ndcgs, _ = self._evaluate_batch(
                    dict_train_pos=self.dataset.dict_train_pos,
                    dict_test_pos=self.dataset.dict_test_pos,
                    userids_batch=userids_batch_test,
                    itemids=itemids_kg,
                    list_k=self.top_ks)

                logging.info('[Evaluation] Epoch {:04d} / {:04d} [{:.1f}s] : hits [{:s}], precision [{:s}], recall [{:s}], ndcg [{:s}]'.format(
                    epoch, self.epochs, time() - time_start_eval,
                    '\t'.join(['%.5f' % h for h in hits]),
                    '\t'.join(['%.5f' % p for p in precisions]),
                    '\t'.join(['%.5f' % r for r in recalls]),
                    '\t'.join(['%.5f' % n for n in ndcgs]),
                ))

                # consider: which indicator should be used?
                list_hit.append(hits[0])
                list_precision.append(precisions[0])
                list_recall.append(recalls[0])
                list_ndcg.append(ndcgs[0])

                is_best, should_stop = is_best_epoch(
                    list_recall, epoch, self.interval_evaluate, self.stopping_steps)
                if is_best:
                    self.best_epoch = epoch
                    self.best_model = self.model
                    self.best_attentions = list(
                        self.graph_train.edata['attention'].cpu().numpy().copy())
                if should_stop:
                    break

        # NOTE: to use best_model for prediction
        self.model = self.best_model
        logging.info('[Finish Training] Epoch {:04d} / {:04d} [{:.1f}s]'.format(
            epoch, self.epochs, time() - time_start_train))

    def _evaluate_batch(
        self, dict_train_pos: UserItems, dict_test_pos: UserItems, userids_batch: List[th.Tensor], itemids: th.Tensor, list_k: List[int], is_predict: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[int, List[int]]]:
        self.model = self.model.eval()

        with th.no_grad():
            self.graph_train.edata['attention'] = self.model.compute_attention(
                self.graph_train)

        hit, precision, recall, ndcg = [], [], [], []
        rec_items = dict()
        n_users = 0
        with th.no_grad():
            for userids in userids_batch:
                # enum で predict を switch する？
                cf_scores_batch = self.model.predict(
                    self.graph_train, userids, itemids)  # (n_batch_users, n_eval_items)

                hits_batch, precision_batch, recall_batch, ndcg_batch, rec_items_batch = compute_metrics(
                    cf_scores_batch.cpu(), dict_train_pos, dict_test_pos,
                    userids.cpu().numpy(), itemids.cpu().numpy(), list_k, is_predict)

                hit.append(hits_batch)
                precision.append(precision_batch)
                recall.append(recall_batch)
                ndcg.append(ndcg_batch)
                n_users += len(userids)

                if is_predict:
                    for i, userid in enumerate(userids):
                        rec_items[int(userid)] = list(rec_items_batch[i])

        hits, precisions, recalls, ngcds = [], [], [], []
        for i in range(len(list_k)):
            hits.append(np.sum([h[i] for h in hit]) / n_users)
            precisions.append(np.sum([p[i] for p in precision]) / n_users)
            recalls.append(np.sum([r[i] for r in recall]) / n_users)
            ngcds.append(np.sum([n[i] for n in ndcg]) / n_users)

        return hits, precisions, recalls, ngcds, rec_items

    def save(self, name_data: str, name_model: str, uniqid: str) -> None:
        path_model = generate_path_model_file(
            name_data, name_model, self.best_epoch, uniqid)
        ensure_file(path_model)
        # NOTE: avoid error when load gpu model from cpu environment
        th.save(self.best_model.to(th.device('cpu')).state_dict(), path_model)

        path_att = generate_path_att_file(
            name_data, name_model, self.best_epoch, uniqid)
        ensure_file(path_att)
        pd.DataFrame({
            # srcids
            'head': self.graph_train.edges()[0].cpu().numpy().copy().astype(np.int32),
            # relation
            'relation': self.graph_train.edata['type'].cpu().numpy().copy().astype(np.int32),
            # dstids
            'tail': self.graph_train.edges()[1].cpu().numpy().copy().astype(np.int32),
            'att': np.array([att[0] for att in self.best_attentions]).astype(np.float32),  # type: ignore
        }).to_csv(path_att, sep=' ', index=False)

    def predict(self, pretrain_path: str = '') -> UserItems:
        if pretrain_path != '':
            self.model = self.load_pretrained_model(self.model, pretrain_path)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model.to(device)

        # ids
        userids_batch_test = [th.LongTensor(self.dataset.uniq_userids[i:i + 10000]).to(
            device) for i in range(0, self.dataset.n_users, 10000)]
        itemids_kg = th.arange(self.dataset.n_items, dtype=th.long).to(
            device) + self.dataset.n_users

        time_start_eval = time()
        hits, precisions, recalls, ndcgs, dict_rec_items = self._evaluate_batch(
            dict_train_pos=self.dataset.dict_train_pos,
            dict_test_pos=self.dataset.dict_test_pos,
            userids_batch=userids_batch_test,
            itemids=itemids_kg,
            list_k=self.top_ks,
            is_predict=True)

        logging.info('[Prediction] [{:.1f}s] : hits [{:s}], precision [{:s}], recall [{:s}], ndcg [{:s}]'.format(
            time() - time_start_eval,
            '\t'.join(['%.5f' % h for h in hits]),
            '\t'.join(['%.5f' % p for p in precisions]),
            '\t'.join(['%.5f' % r for r in recalls]),
            '\t'.join(['%.5f' % n for n in ndcgs]),
        ))

        return dict_rec_items
