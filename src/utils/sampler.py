import random
from typing import List, Tuple

import numpy as np

from .types import Triplet, UserItems


def generate_cf_batch(dict_user_items: UserItems, batch_size: int, n_shift: int, n_items: int) -> Tuple[List[int], List[int], List[int]]:
    exist_users = list(dict_user_items.keys())
    if batch_size <= len(exist_users):
        batch_user = random.sample(exist_users, batch_size)
    else:
        batch_user = [random.choice(exist_users) for _ in range(batch_size)]

    batch_pos_item, batch_neg_item = [], []
    for u in batch_user:
        batch_pos_item += sample_pos_items_for_u(
            dict_user_items, u, 1, n_shift)
        batch_neg_item += sample_neg_items_for_u(
            dict_user_items, u, 1, n_shift, n_items)
    return batch_user, batch_pos_item, batch_neg_item


def sample_pos_items_for_u(dict_user_items: UserItems, userid: int, n_sample_pos_items: int, n_shift: int) -> List[int]:
    # itemids in cf data should be shifted when user and item embedds are included in the same param.
    pos_items = list(map(lambda x: x + n_shift, dict_user_items[userid]))
    n_pos_items = len(pos_items)
    sample_pos_items: List[int] = []
    # consider: not need while & n_sample_pos_items
    while True:
        if len(sample_pos_items) == n_sample_pos_items:
            break
        pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_item_id = pos_items[pos_item_idx]
        if pos_item_id not in sample_pos_items:
            sample_pos_items.append(pos_item_id)
    return sample_pos_items


def sample_neg_items_for_u(dict_user_items: UserItems, userid: int, n_sample_neg_items: int, n_shift: int, n_items: int) -> List[int]:
    # itemids in cf data should be shifted when user and item embedds are included in the same param.
    pos_items = list(map(lambda x: x + n_shift, dict_user_items[userid]))
    sample_neg_items: List[int] = []
    while True:
        if len(sample_neg_items) == n_sample_neg_items:
            break
        neg_item_id = np.random.randint(low=0, high=n_items, size=1)[0]
        if (neg_item_id not in pos_items) and (neg_item_id not in sample_neg_items):
            sample_neg_items.append(neg_item_id)
    return sample_neg_items


def generate_kg_batch(kg_dict: Triplet, batch_size: int, n_entities_ckg_all: int) -> Tuple[List[int], List[int], List[int], List[int], List[float]]:
    exist_heads = list(kg_dict.keys())
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation: List[int] = []
    batch_pos_tail: List[int] = []
    batch_neg_tail: List[int] = []
    batch_pos_epsilon: List[float] = []
    for h in batch_head:
        relation, pos_tail, pos_epsilon = sample_pos_triples_for_h(
            kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail
        batch_pos_epsilon += pos_epsilon

        neg_tail = sample_neg_triples_for_h(
            kg_dict, h, relation[0], 1, n_entities_ckg_all)
        batch_neg_tail += neg_tail
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail, batch_pos_epsilon


def sample_pos_triples_for_h(kg_dict: Triplet, head: int, n_sample_pos_triples: int) -> Tuple[List[int], List[int], List[float]]:
    pos_triples = kg_dict[head]
    prior_dists = np.array(pos_triples)[:, 2]
    n_pos_triples = len(pos_triples)

    sample_relations: List[int] = []
    sample_pos_tails: List[int] = []
    sample_pos_epsilons: List[float] = []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break
        pos_triple_idx = random.choices(
            np.arange(n_pos_triples), k=1, weights=prior_dists)[0]
        tail = pos_triples[pos_triple_idx][0]
        relation = pos_triples[pos_triple_idx][1]
        epsilon = pos_triples[pos_triple_idx][2]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
            sample_pos_epsilons.append(epsilon)
    return sample_relations, sample_pos_tails, sample_pos_epsilons


def sample_neg_triples_for_h(kg_dict: Triplet, head: int, relation: int, n_sample_neg_triples: int, n_entities_ckg_all: int) -> List[int]:
    pos_triples = kg_dict[head]
    sample_neg_tails: List[int] = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break
        tail = np.random.randint(low=0, high=n_entities_ckg_all, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails
