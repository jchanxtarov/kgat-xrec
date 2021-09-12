import enum
from typing import Dict, List, Tuple

# dataset
Triplet = Dict[int, List[Tuple[int, int, float]]]
UserItems = Dict[int, List[int]]


class KgatTrainType(enum.Enum):
    COMP_LOSS_CF = enum.auto()
    COMP_LOSS_KG = enum.auto()
    PRED = enum.auto()


class ModelType(enum.Enum):
    KGAT = "kgat"
    KGATpls = "kgat+"
    BPRMF = "bprmf"
    NFM = "nfm"
    CFKG = "cfkg"
