import numpy as np
import torch as th


def compute_metrics(matrix_scores, dict_train_pos, dict_test_pos, userids, itemids, list_k, is_predict=False):
    matrix_test_pos = np.zeros([len(userids), len(itemids)], dtype=np.float32)
    for i, userid in enumerate(userids):
        train_pos_itemids = dict_train_pos[userid]
        test_pos_itemids = dict_test_pos[userid]
        # already bought in train data
        matrix_scores[i][train_pos_itemids] = 0.0
        matrix_test_pos[i][test_pos_itemids] = 1.0  # ground truth in test data

    try:
        _, idx = th.sort(matrix_scores.cuda(), descending=True)
    except:
        _, idx = th.sort(matrix_scores, descending=True)

    rec_items = None
    if is_predict:
        k = list_k[-1]
        rec_items = np.argpartition(idx.cpu().numpy(), k)[:, :k]

    # TODO: use th.gather and make here rapid
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    idx = idx.cpu()
    matrix_hit = [matrix_test_pos[i][idx[i]] for i in range(len(userids))]
    matrix_hit = np.array(matrix_hit, dtype=np.float32)

    hits = compute_hit(matrix_hit, list_k)
    precisions = compute_precision(matrix_hit, list_k)
    recalls = compute_recall(matrix_hit, list_k)
    ndcgs = compute_ndcg(matrix_hit, list_k)
    return hits, precisions, recalls, ndcgs, rec_items


def compute_hit(hits, list_k):
    return [np.sum(np.where(hits[:, :k].sum(axis=1) > 0, 1.0, 0.0)) for k in list_k]


def compute_precision(hits, list_k):
    return [np.sum(hits[:, :k].mean(axis=1)) for k in list_k]


def compute_recall(hits, list_k):
    return [np.sum(hits[:, :k].sum(axis=1) / hits.sum(axis=1)) for k in list_k]

# see also: https://en.wikipedia.org/wiki/Discounted_cumulative_gain


def compute_ndcg(hits, list_k, method=1):
    list_ndcg = []
    for k in list_k:
        # numerator
        dcg = np.sum(((2 ** hits[:, :k]) - 1) /
                     np.log2(np.arange(2, k + 2)), axis=1)
        # denominator
        sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
        ideal_dcg = np.sum(((2 ** sorted_hits_k) - 1) /
                           np.log2(np.arange(2, k + 2)), axis=1)
        ideal_dcg[ideal_dcg == 0] = np.inf
        list_ndcg.append(np.sum(dcg / ideal_dcg))
    return list_ndcg
