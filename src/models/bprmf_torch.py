from typing import List

import torch as th
from torch import nn
from torch.nn import functional as F
from utils.loaders.bprmf import BprmfDataset


class BPRMF(nn.Module):
    def __init__(
            self,
            dataset: BprmfDataset,
            dim_entity_embed: int,
            list_reg_learning_rates: List[float]
    ) -> None:
        super(BPRMF, self).__init__()

        self.user_embed = nn.Embedding(dataset.n_users, dim_entity_embed)
        nn.init.xavier_uniform_(self.user_embed.weight,
                                gain=nn.init.calculate_gain('relu'))
        self.item_embed = nn.Embedding(dataset.n_items, dim_entity_embed)
        nn.init.xavier_uniform_(self.item_embed.weight,
                                gain=nn.init.calculate_gain('relu'))

        self.reg_learning_rates = list_reg_learning_rates[0]

    def forward(self, *input):
        return self._compute_loss(*input)

    def _compute_loss(self, userids, itemids_pos, itemids_neg):
        user_embed = self.user_embed(userids)  # (batch_size, dim_entity_embed)
        # (batch_size, dim_entity_embed)
        item_embed_pos = self.item_embed(itemids_pos)
        # (batch_size, dim_entity_embed)
        item_embed_neg = self.item_embed(itemids_neg)

        pos_score = th.sum(user_embed * item_embed_pos, dim=1)  # (batch_size)
        neg_score = th.sum(user_embed * item_embed_neg, dim=1)  # (batch_size)

        base_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        base_loss = th.mean(base_loss)

        reg_loss = self._l2_loss(
            user_embed) + self._l2_loss(item_embed_pos) + self._l2_loss(item_embed_neg)
        reg_loss = self.reg_learning_rates * reg_loss
        return base_loss, reg_loss

    def _l2_loss(self, embedd):
        return th.sum(embedd.pow(2).sum(1) / 2.0)

    def predict(self, userids, itemids):
        # (n_eval_users, dim_entity_embed)
        user_embed = self.user_embed(userids)
        # (n_eval_items, dim_entity_embed)
        item_embed = self.item_embed(itemids)
        # (n_eval_users, n_eval_items)
        return th.matmul(user_embed, item_embed.transpose(0, 1))
