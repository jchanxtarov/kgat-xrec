from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn  # type: ignore
from dgl.heterograph import DGLHeteroGraph  # type: ignore
from dgl.view import HeteroEdgeDataView  # type: ignore
from torch.nn.modules.sparse import Embedding
from utils.loaders.kgat import KgatDataset
from utils.types import KgatTrainType


class KGAT(nn.Module):
    def __init__(
        self,
        dataset: KgatDataset,
        dim_entity_embed: int,
        dim_relation_embed: int,
        list_reg_learning_rates: List[float],
        list_dim_conv_layer: List[int],
        list_size_dropout: List[float],
        learning_rate: float,
    ) -> None:
        super(KGAT, self).__init__()

        self.n_relations = dataset.n_relations
        self.n_entities_ckg_all = dataset.n_entities_ckg_all
        self.learning_rate = learning_rate
        self.list_reg_learning_rates = list_reg_learning_rates
        list_dim_layers = [dim_entity_embed] + list_dim_conv_layer

        # torch parameters
        # self.register_parameter(name="trans_w", param=nn.Parameter(th.Tensor(self.n_relations, dim_entity_embed, dim_relation_embed)))
        self.trans_w = nn.Parameter(
            th.Tensor(self.n_relations, dim_entity_embed, dim_relation_embed))
        nn.init.xavier_uniform_(self.trans_w, gain=nn.init.calculate_gain(
            'relu'))  # consider: 1.0 or relu

        # entity_embed: users -> items -> entities
        self.entity_embed = nn.Embedding(
            dataset.n_entities_ckg_all, dim_entity_embed)
        nn.init.xavier_uniform_(self.entity_embed.weight, gain=nn.init.calculate_gain(
            'relu'))  # consider: 1.0 or relu
        self.relation_embed = nn.Embedding(
            self.n_relations, dim_relation_embed)
        nn.init.xavier_uniform_(self.relation_embed.weight, gain=nn.init.calculate_gain(
            'relu'))  # consider: 1.0 or relu

        self.aggregator_layers = nn.ModuleList()
        for k in range(len(list_dim_conv_layer)):
            self.aggregator_layers.append(BiInteractionAggregator(
                list_dim_layers[k], list_dim_layers[k + 1], list_size_dropout[k]))

    def forward(self, mode: KgatTrainType, *input) -> Tuple[th.Tensor, th.Tensor]:
        if mode == KgatTrainType.COMP_LOSS_CF:
            return self._compute_loss_cf(*input)
        elif mode == KgatTrainType.COMP_LOSS_KG:
            return self._compute_loss_kg(*input)
        else:
            raise NotImplementedError()

    # see also:
    # - https://github.com/xiangwang1223/knowledge_graph_attention_network/blob/master/Model/Main.py#L56
    # - https://github.com/xiangwang1223/knowledge_graph_attention_network/blob/master/Model/utility/loader_kgat.py#L22
    def initialize_attention(self, graph: DGLHeteroGraph) -> th.Tensor:
        graph = graph.local_var()
        for r in range(self.n_relations):
            edges = graph.filter_edges(lambda edge: edge.data['type'] == r)
            graph.apply_edges(self._si_norm_lap, edges)

        return graph.edata['out']

    def _si_norm_lap(self, edges: HeteroEdgeDataView) -> Dict[str, th.Tensor]:
        adj = sp.coo_matrix(([1.0] * len(edges), (edges.src['id'].cpu(), edges.dst['id'].cpu())),
                            shape=(self.n_entities_ckg_all, self.n_entities_ckg_all))
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        return {'out': th.tensor(norm_adj.data, dtype=th.float32).reshape(-1, 1).to(device)}

    def _compute_loss_cf(self, graph: DGLHeteroGraph, userids: th.Tensor, itemids_pos: th.Tensor, itemids_neg: th.Tensor):
        # (n_users + n_entities, cf_concat_dim)
        all_embed = self._cf_embedding(KgatTrainType.COMP_LOSS_CF, graph)
        user_embed = all_embed[userids]  # (batch_size, cf_concat_dim)
        item_embed_pos = all_embed[itemids_pos]  # (batch_size, cf_concat_dim)
        item_embed_neg = all_embed[itemids_neg]  # (batch_size, cf_concat_dim)

        # Eq.(10)
        scores_pos = th.sum(user_embed * item_embed_pos, dim=1)  # (batch_size)
        scores_neg = th.sum(user_embed * item_embed_neg, dim=1)  # (batch_size)

        # Eq.(12)
        # see also: https://github.com/xiangwang1223/knowledge_graph_attention_network/blob/master/Model/KGAT.py#L220
        base_loss = F.softplus(-1.0 * (scores_pos - scores_neg))
        base_loss = th.sum(base_loss)  # consider: test
        reg_loss = self._l2_loss(
            user_embed) + self._l2_loss(item_embed_pos) + self._l2_loss(item_embed_neg)
        reg_loss = self.list_reg_learning_rates[0] * reg_loss
        return base_loss, reg_loss

    def _cf_embedding(self, mode: KgatTrainType, graph: DGLHeteroGraph) -> th.Tensor:
        graph = graph.local_var()
        ego_embed = self.entity_embed(graph.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, graph, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)  # same as github
            all_embed += [norm_embed]

        # Eq.(11)
        # (n_users + n_entities, cf_concat_dim)
        return th.cat(all_embed, dim=1)

    def compute_attention(self, graph: DGLHeteroGraph) -> th.Tensor:
        graph = graph.local_var()
        for r in range(self.n_relations):
            edges = graph.filter_edges(lambda edge: edge.data['type'] == r)
            self.trans_w_r = self.trans_w[r]
            # see also: https://docs.dgl.ai/en/0.6.x/generated/dgl.DGLGraph.apply_edges.html#dgl.DGLGraph.apply_edges
            graph.apply_edges(self._compute_attention, edges)

        return self._normalize_attention(graph)

    def _compute_attention(self, edges: HeteroEdgeDataView) -> Dict[str, th.Tensor]:
        r_embedd = self.relation_embed(edges.data['type'])  # (1, dim_relation)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        epsilons = edges.data['weight'].to(device)  # (1, dim_relation)
        # see also: https://docs.dgl.ai/en/0.6.x/api/python/udf.html
        # (n_edge, dim_relation)
        wr_h = th.matmul(self.entity_embed(edges.src['id']), self.trans_w_r)
        # (n_edge, dim_relation)
        wr_t = th.matmul(self.entity_embed(edges.dst['id']), self.trans_w_r)

        # Eq.(5)
        att = epsilons.reshape(-1, 1) * th.bmm(wr_t.unsqueeze(1), th.tanh(wr_h + r_embedd).unsqueeze(2)).squeeze(-1)  # (n_edge, 1)
        return {'attention': att}

    def _normalize_attention(self, graph: DGLHeteroGraph) -> th.Tensor:
        graph = graph.local_var()
        graph.edata['out'] = th.exp(graph.edata['attention'])
        # see also: https://docs.dgl.ai/en/0.4.x/generated/dgl.function.sum.html
        graph.update_all(fn.copy_edge('out', 'm'), fn.sum('m', 'sum_out'))

        # Eq.(6)
        graph.apply_edges(fn.e_div_v('out', 'sum_out', 'out')
                          )  # (exp_att / sum_exp_att)
        return graph.edata['out']

    def _compute_loss_kg(self, h: th.Tensor, r: th.Tensor, pos_t: th.Tensor, neg_t: th.Tensor, eps: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        r_embed = self.relation_embed(r)  # (batch_size, dim_relation)
        # (batch_size, dim_entity_embed, dim_relation)
        trans_w_r = self.trans_w[r]

        h_embed = self.entity_embed(h)  # (batch_size, dim_entity_embed)
        # (batch_size, dim_entity_embed)
        t_embed_pos = self.entity_embed(pos_t)
        # (batch_size, dim_entity_embed)
        t_embed_neg = self.entity_embed(neg_t)

        wr_h = th.bmm(h_embed.unsqueeze(1), trans_w_r).squeeze(
            1)  # (batch_size, dim_relation)
        wr_pos_t = th.bmm(t_embed_pos.unsqueeze(1), trans_w_r).squeeze(
            1)  # (batch_size, dim_relation)
        wr_neg_t = th.bmm(t_embed_neg.unsqueeze(1), trans_w_r).squeeze(
            1)  # (batch_size, dim_relation)

        # Eq.(3)
        scores_pos = th.sum(
            th.square(wr_h + r_embed - wr_pos_t), dim=1)  # (batch_size)
        scores_neg = th.sum(
            th.square(wr_h + r_embed - wr_neg_t), dim=1)  # (batch_size)

        # Eq.(2)
        # see also: https://github.com/xiangwang1223/knowledge_graph_attention_network/blob/master/Model/KGAT.py#L241
        base_loss = eps * F.softplus(-1.0 * (scores_neg - scores_pos))
        base_loss = th.sum(base_loss)
        reg_loss = self._l2_loss(wr_h) + self._l2_loss(r_embed) + \
            self._l2_loss(wr_pos_t) + self._l2_loss(wr_neg_t)
        reg_loss = self.list_reg_learning_rates[1] * reg_loss
        return base_loss, reg_loss

    def _l2_loss(self, embedd: th.Tensor) -> th.Tensor:
        return th.sum(embedd.pow(2).sum(1) / 2.0)  # consider: test

    def predict(self, graph: DGLHeteroGraph, userids: th.Tensor, itemids: th.Tensor) -> th.Tensor:
        # (n_users + n_entities, cf_concat_dim)
        all_embed = self._cf_embedding(KgatTrainType.PRED, graph)
        user_embed = all_embed[userids]  # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[itemids]  # (n_eval_items, cf_concat_dim)

        # Eq.(10)
        # (n_eval_users, n_eval_items)
        return th.matmul(user_embed, item_embed.T)


class BiInteractionAggregator(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, prob: float):
        super(BiInteractionAggregator, self).__init__()

        # same as github: (weight, bias)
        self.w1 = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.w1.weight, gain=nn.init.calculate_gain(
            'relu'))  # consider: 1.0 or relu
        # see also: https://github.com/pytorch/pytorch/issues/3418
        nn.init.constant(self.w1.bias, 0)
        self.w2 = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.w2.weight, gain=nn.init.calculate_gain(
            'relu'))  # consider: 1.0 or relu
        nn.init.constant(self.w2.bias, 0)

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=prob)

    def forward(self, mode: KgatTrainType, graph: DGLHeteroGraph, entity_embed: Embedding) -> Embedding:
        graph = graph.local_var()
        graph.ndata['node'] = entity_embed

        # Eq.(9)
        if mode == KgatTrainType.PRED:
            graph.update_all(fn.u_mul_e('node', 'attention', 'side'), lambda nodes: {
                             'ego': th.sum(nodes.mailbox['side'], 1)})
        else:
            graph.update_all(fn.u_mul_e('node', 'attention',
                                        'side'), fn.sum('side', 'ego'))

        # Eq.(8)
        # w1: Linear(weight + bias)
        sum_embedd = self.activation(
            self.w1(graph.ndata['node'] + graph.ndata['ego']))
        # w2: Linear(weight + bias)
        bi_embedd = self.activation(
            self.w2(graph.ndata['node'] * graph.ndata['ego']))
        ego_embedd = sum_embedd + bi_embedd
        ego_embedd = self.dropout(ego_embedd)
        return ego_embedd
