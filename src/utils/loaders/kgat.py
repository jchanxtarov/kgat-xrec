import logging
from collections import defaultdict
from dataclasses import dataclass

import dgl  # type: ignore
import numpy as np
import torch as th
from dgl.heterograph import DGLHeteroGraph  # type: ignore

from ..types import Triplet
from .base import BaseDataset


@dataclass
class KgatDataset(BaseDataset):
    graph_train: DGLHeteroGraph
    graph_test: DGLHeteroGraph
    dict_h_tre_all: Triplet

    def __init__(
        self,
        # for base loader
        dataset: str,
        kg_file: str,
        use_kgat_plus: bool,
        min_edge_prob: float,
        use_user_attribute: bool,
    ) -> None:
        super().__init__(
            name_dataset=dataset,
            kg_file=kg_file,
            use_kg=True,
            use_kgat_plus=use_kgat_plus,
            min_edge_prob=min_edge_prob,
            use_user_attribute=use_user_attribute,
        )

        self._preprocessing(use_user_attribute)

    def _preprocessing(self, use_user_attribute: bool) -> None:
        self._generate_ckg(use_user_attribute)
        self._generate_dict_h_tre_all()

    def _generate_ckg(self, use_user_attribute: bool) -> None:

        def _generate_graph(uids: np.ndarray, iids: np.ndarray, n_cf: int) -> DGLHeteroGraph:
            # 1. cf2kg
            """memo
                about nodeid
                    - 0 ~ + n_users: userid
                    - ~ + n_items: itemid
                    - ~ + n_enities: entityid
            """
            uids_cf2kg = uids
            iids_cf2kg = iids
            iids_cf2kg += self.n_users
            r_cf2kg = th.zeros(n_cf, dtype=th.long)  # 0: buy
            eps_cf2kg = th.ones(n_cf, dtype=th.long)

            # 2. remap kg
            hids_kg = self.arr_kg[:, 0]
            rids_kg = self.arr_kg[:, 1] + 1  # 0: buy
            tids_kg = self.arr_kg[:, 2]
            eps_kg = self.arr_kg[:, 3]
            # if True, it must be remapped in original kg data.
            if not use_user_attribute:
                hids_kg += self.n_users
                tids_kg += self.n_users

            # 3. concat cf2kg and kg
            hids_ckg = np.append(uids_cf2kg, hids_kg)
            tids_ckg = np.append(iids_cf2kg, tids_kg)
            rids_ckg = np.append(r_cf2kg, rids_kg)
            eps_ckg = np.append(eps_cf2kg, eps_kg)

            # 4. apply dgl
            n_nodes = self.n_entities_ckg_all
            data = (hids_ckg, tids_ckg)
            g = dgl.graph(data, num_nodes=n_nodes)
            g.ndata['id'] = th.arange(n_nodes, dtype=th.long)
            g.edata['weight'] = th.tensor(eps_ckg, dtype=th.float32)

            # 5. concat reversed ckg
            ckg = dgl.add_reverse_edges(g, copy_edata=True)
            # reverse relation and define as new relation
            rids_ckg = np.append(rids_ckg, rids_ckg + np.unique(rids_ckg).size)
            ckg.edata['type'] = th.LongTensor(rids_ckg)
            ckg.readonly()

            return ckg

        self.graph_train = _generate_graph(self.arr_train[:, 0],
                                           self.arr_train[:, 1], self.n_train)
        self.graph_test = _generate_graph(self.arr_test[:, 0],
                                          self.arr_test[:, 1], self.n_test)

    def _generate_dict_h_tre_all(self) -> None:
        heads = list(self.graph_train.edges()[0].numpy().copy())
        tails = list(self.graph_train.edges()[1].numpy().copy())
        relations = list(self.graph_train.edata['type'].numpy().copy())
        epsilons = list(self.graph_train.edata['weight'].numpy().copy())
        self.epsilons = epsilons

        default_dict_h_tre_all = defaultdict(list)
        for i in range(len(heads)):
            default_dict_h_tre_all[heads[i]].append(
                (tails[i], relations[i], epsilons[i]))
        self.dict_h_tre_all = dict(default_dict_h_tre_all)

    def logging_statistics(self) -> None:
        logging.info("n_train:            {}".format(self.n_train))
        logging.info("n_test:             {}".format(self.n_test))
        logging.info("n_users:            {}".format(self.n_users))
        logging.info("n_items:            {}".format(self.n_items))
        logging.info("n_relations:        {}".format(self.n_relations))
        logging.info("n_entities:         {}".format(self.n_entities))
        logging.info("n_entities_ckg_all: {}".format(self.n_entities_ckg_all))
        logging.info("n_triplet_kg:       {}".format(self.n_triplet_kg))
