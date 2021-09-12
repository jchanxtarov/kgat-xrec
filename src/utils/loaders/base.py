from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd  # type: ignore

from ..types import Triplet, UserItems


@dataclass
class BaseDataset(metaclass=ABCMeta):
    n_train: int
    n_test: int
    n_users: int
    n_items: int
    n_entities: int
    n_relations: int
    n_entities_ckg_all: int
    n_triplet_kg: int
    dict_train_pos: UserItems
    dict_test_pos: UserItems
    dict_h_tre: Triplet
    arr_kg: np.ndarray
    arr_train: np.ndarray
    arr_test: np.ndarray
    uniq_userids: List[int]

    def __init__(
        self,
        name_dataset: str,
        kg_file: str = "",
        use_kg: bool = False,
        use_kgat_plus: bool = False,
        min_edge_prob: float = 1.0e-2,
        use_user_attribute: bool = False,
    ) -> None:
        path_train_cf = "./src/datasets/{}/train.txt".format(name_dataset)
        path_test_cf = "./src/datasets/{}/test.txt".format(name_dataset)
        path_kg = "./src/datasets/{0}/{1}".format(name_dataset, kg_file)

        # read files
        self.arr_train, self.dict_train_pos = self._load_cf(path_train_cf)
        self.arr_test, self.dict_test_pos = self._load_cf(path_test_cf)
        self.n_train = len(self.arr_train)
        self.n_test = len(self.arr_test)

        # get statics
        self.uniq_userids = sorted(list(self.dict_train_pos.keys()))
        # must be same n test & train users
        self.n_users = max(self.uniq_userids) + 1
        self.n_items = max(
            max(self.arr_train[:, 1]), max(self.arr_test[:, 1])) + 1

        if use_kg:
            self.arr_kg = self._load_kg(path_kg, use_kgat_plus, min_edge_prob)
            self.n_triplet_kg = len(self.arr_kg)
            # add buy(+1) -> reverse(*2)
            self.n_relations = 2 * ((max(self.arr_kg[:, 1]) + 1) + 1)
            self.n_entities = max(max(self.arr_kg[:, 0]), max(
                self.arr_kg[:, 2])) - self.n_items + 1
            if use_user_attribute:
                self.n_entities -= self.n_users
            self.n_entities_ckg_all = self.n_users + self.n_items + self.n_entities

    # TODO: easy checker insetead of test module
    # private 変数(__xxx)を擬似的にしか作成できないので，書き換え後に実行されるチェッカー(post_init)があった方が良さそう
    # def __post_init__(self) -> None:
    #     if not check_inner_vals():
    #         raise ValueError("n_items_entities must be a n_items+n_entities")

    def _load_cf(self, path_dataset: str) -> Tuple[np.ndarray, UserItems]:
        dict_uid_iids = dict()
        list_uid_iid = list()

        lines = open(path_dataset, "r").readlines()
        for line in lines:
            tmps = line.strip()
            inters = [int(i) for i in tmps.split(" ")]

            userid, itemids = inters[0], inters[1:]
            itemids = list(set(itemids))

            for itemid in itemids:
                list_uid_iid.append([userid, itemid])

            if len(itemids) > 0:
                dict_uid_iids[userid] = itemids
        return np.array(list_uid_iid), dict_uid_iids

    def _load_kg(
        self,
        path_dataset: str,
        use_kgat_plus: bool,
        min_edge_prob: float,
    ) -> np.ndarray:
        if use_kgat_plus:
            # consider: ここだけ抽象メソッドにする？
            df_kg = pd.read_table(path_dataset, header=None, delimiter=",")
            df_kg = df_kg[df_kg[3] > min_edge_prob].reset_index(drop=True)
            arr_kg = np.array(df_kg.values[:, 0:3], dtype=np.int32)
            epsilons = np.array(df_kg.values[:, 3], dtype=np.float32)
            del df_kg
        else:
            arr_kg = np.loadtxt(path_dataset, dtype=np.int32, delimiter=' ')
            epsilons = np.ones(len(arr_kg), dtype=np.float32)

        default_dict_h_tre = defaultdict(list)

        for i, (head, relation, tail) in enumerate(arr_kg):
            default_dict_h_tre[head].append((tail, relation, epsilons[i]))

        self.dict_h_tre = dict(default_dict_h_tre)

        return np.insert(arr_kg, 3, epsilons, axis=1)

    # TODO: remove comment out
    # @abstractmethod
    def logging_statistics(self) -> None:
        raise NotImplementedError()
