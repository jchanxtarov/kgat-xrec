from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

import torch as th
import torch.nn as nn
from utils.loaders.base import BaseDataset
from utils.types import UserItems

Dataset = TypeVar('Dataset', bound=BaseDataset)


class BasePredictor(Generic[Dataset], metaclass=ABCMeta):

    @abstractmethod
    def load(self, dataset: Dataset) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, name_data: str, name_model: str, uniqid: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, pretrain_path: str) -> UserItems:
        raise NotImplementedError

    # TODO: avoid Any
    def load_pretrained_model(self, model: nn.Module, path: str) -> Any:
        model.load_state_dict(th.load(path, map_location=th.device('cpu')))
        return model
