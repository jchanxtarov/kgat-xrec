import logging
import os
from logging import DEBUG, NOTSET, FileHandler, Formatter
from typing import Tuple

from .types import UserItems


def ensure_file(path: str) -> None:
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def generate_path_log_file(name_data: str, name_model: str, uniqid: str) -> str:
    return "./src/outputs/{0}/log/{1}_{2}.log".format(name_data, name_model, uniqid)


def generate_path_model_file(name_data: str, name_model: str, epoch: int, uniqid: str) -> str:
    return "./src/outputs/{0}/pretrain/{1}_epoch{2}_{3}.pth".format(name_data, name_model, epoch, uniqid)


def generate_path_att_file(name_data: str, name_model: str, epoch: int, uniqid: str) -> str:
    return "./src/outputs/{0}/result/attention_{1}_epoch{2}{3}.txt".format(name_data, name_model, epoch, uniqid)


def generate_path_pretrain_file(name_data: str, filename: str) -> str:
    return "./src/outputs/{0}/pretrain/{1}".format(name_data, filename)


def generate_path_recitems_file(name_data: str, name_model: str, uniqid: str) -> str:
    return "./src/outputs/{0}/result/recitems_{1}_{2}.txt".format(name_data, name_model, uniqid)


def set_logging(path: str, save_log: bool) -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter("%(message)s"))
    handlers = [stream_handler]
    path_log = "None"

    if save_log:
        path_log = os.path.join(path)
        file_handler = FileHandler(path_log)
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(
            Formatter("%(asctime)s @ %(name)s [%(levelname)s] %(funcName)s: %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(level=NOTSET, handlers=handlers)
    logging.debug(
        "Success to set logging & generate logfile where {}.".format(path_log))


def is_best_epoch(list_recall, current_epoch, interval_evaluate, stopping_steps) -> Tuple[bool, bool]:
    best_idx = list_recall.index(max(list_recall))
    best_epoch = (best_idx + 1) * interval_evaluate

    is_best = False
    should_stop = False
    if current_epoch >= best_epoch:
        is_best = True
    if best_epoch >= stopping_steps:
        should_stop = True
    return is_best, should_stop


def save_recommended_items_list(path: str, dict_user_items: UserItems) -> None:
    f = open(path, 'w')
    for k, v in dict_user_items.items():
        f.write(str(k) + ',' + str(v) + '\n')
    f.close()
