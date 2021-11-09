import argparse
from datetime import datetime as dt

from utils.types import ModelType


def parse_args():
    parser = argparse.ArgumentParser()

    # about dataset
    parser.add_argument(
        '--seed',
        type=int,
        default=2020,
        help='Set random seed.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='zr-sample',
        help='Select the target dataset.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=ModelType.KGATpls.value,
        choices=[model.value for model in ModelType],
        help='Select model.'
    )
    parser.add_argument(
        '--kg_file',
        type=str,
        default='kg_plsa6.txt',  # TODO: kg_test.txt
        help='Select the target dataset.',
    )
    parser.add_argument(
        '--min_edge_prob',
        type=float,
        default=1e-2,
        help='Set minimum value of relation strength (epsilon.))'
    )
    parser.add_argument(
        '--use_user_attribute',
        action='store_true',
        help='Wheather including users side information in kg data.'
    )

    # about predictor
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Set max epochs.'
    )
    parser.add_argument(
        '--dim_entity_embed',
        type=int,
        default=64,
        help='Set dim_entity_embed.'
    )
    parser.add_argument(
        '--dim_relation_embed',
        type=int,
        default=64,
        help='Set dim_relation_embed.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Set batch size.'
    )
    parser.add_argument(
        '--list_reg_learning_rates',
        nargs='?',
        default='[1e-5, 1e-5]',
        help='Set reglarization learning rates for [cf, kge].'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Set learning_rate.'
    )
    parser.add_argument(
        '--list_dim_conv_layer',
        nargs='?',
        default='[64, 32, 16]',
        help='Set dim_conv_layer as list.'
    )
    parser.add_argument(
        '--list_size_dropout',
        nargs='?',
        default='[0.1, 0.1, 0.1]',
        help='Set list_size_dropout as list.'
    )
    parser.add_argument(
        '--top_ks',
        nargs='?',
        default='[20, 60, 100]',
        help='Set top_ks as list.'
    )
    parser.add_argument(
        '--interval_evaluate',
        type=int,
        default=1,
        help='Set interval_evaluate.'
    )
    parser.add_argument(
        '--stopping_steps',
        type=int,
        default=300,
        help='Set stopping_steps.'
    )

    # about experiment condition
    parser.add_argument(
        '--save_log',
        action='store_true',
        help='Whether saving log.'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Whether saving best model & attention.'
    )

    # for prediction
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Whether to predict.'
    )
    parser.add_argument(
        '--use_pretrain',
        action='store_true',
        help='Whether to use pretrained parapemters.'
    )
    parser.add_argument(
        '--save_recommended_items',
        action='store_true',
        help='Whether to save recommendation items list.'
    )
    parser.add_argument(
        '--pretrain_file',
        type=str,
        # required=False  # TODO: default:
        default='kgat+_epoch0_20210830-111759.pth',
        help='Select the target dataset.',
    )

    args = parser.parse_args()
    args.dt_now = dt.now().strftime("%Y%m%d-%H%M%S")
    args.list_reg_learning_rates = eval(args.list_reg_learning_rates)
    args.list_dim_conv_layer = eval(args.list_dim_conv_layer)
    args.list_size_dropout = eval(args.list_size_dropout)
    args.top_ks = eval(args.top_ks)

    return args
