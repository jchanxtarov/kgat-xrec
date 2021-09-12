import logging
import random
from typing import Any

import numpy as np
import torch as th

from models.bprmf import BprmfPredictor
from models.kgat import KgatPredictor
from utils.helper import (ensure_file, generate_path_log_file,
                          generate_path_pretrain_file,
                          generate_path_recitems_file,
                          save_recommended_items_list, set_logging)
from utils.loaders.bprmf import BprmfDataset
from utils.loaders.kgat import KgatDataset
from utils.parser import parse_args
from utils.types import ModelType


def initialize(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    path_logfile = generate_path_log_file(
        args.dataset, args.model, args.dt_now)
    if args.save_log:
        ensure_file(path_logfile)
    set_logging(path_logfile, args.save_log)


if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    logging.info(args)
    dataset: Any  # TODO: avoid Any
    model: Any  # TODO: avoid Any

    if ModelType.KGAT.value in args.model:
        use_kgat_plus = False
        if args.model == ModelType.KGATpls.value:
            use_kgat_plus = True

        dataset = KgatDataset(
            args.dataset,
            args.kg_file,
            use_kgat_plus,
            args.min_edge_prob,
            args.use_user_attribute,
        )
        dataset.logging_statistics()

        model = KgatPredictor(
            epochs=args.epochs,
            dim_entity_embed=args.dim_entity_embed,
            dim_relation_embed=args.dim_relation_embed,
            batch_size=args.batch_size,
            list_reg_learning_rates=args.list_reg_learning_rates,
            learning_rate=args.learning_rate,
            list_dim_conv_layer=args.list_dim_conv_layer,
            list_size_dropout=args.list_size_dropout,
            top_ks=args.top_ks,
            interval_evaluate=args.interval_evaluate,
            stopping_steps=args.stopping_steps,
        )

    if args.model == ModelType.BPRMF.value:
        dataset = BprmfDataset(args.dataset)
        dataset.logging_statistics()

        model = BprmfPredictor(
            epochs=args.epochs,
            dim_entity_embed=args.dim_entity_embed,
            batch_size=args.batch_size,
            list_reg_learning_rates=args.list_reg_learning_rates,
            learning_rate=args.learning_rate,
            top_ks=args.top_ks,
            interval_evaluate=args.interval_evaluate,
            stopping_steps=args.stopping_steps,
        )

    model.load(dataset)

    if not args.use_pretrain:
        model.train()

        if args.save_model:
            model.save(args.dataset, args.model, args.dt_now)

    if args.predict:
        path_pretrain = ''
        if args.use_pretrain and args.pretrain_file:
            path_pretrain = generate_path_pretrain_file(
                args.dataset, args.pretrain_file)
        rec_items = model.predict(path_pretrain)

        if args.save_recommended_items:
            path_items = generate_path_recitems_file(
                args.dataset, args.model, args.dt_now)
            ensure_file(path_items)
            save_recommended_items_list(path_items, rec_items)
