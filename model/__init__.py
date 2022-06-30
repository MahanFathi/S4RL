from ml_collections import FrozenConfigDict
from .layers import *
from .seqs import BatchStackedModel


def get_layer_cls(name: str, N: int):
    return globals()[name + "Init"](N)


def create_model(
        cfg: FrozenConfigDict,
        d_output: int,
        seq_len: int,
        decode: bool = False, # True = RNN (Testing), False = CNN (Training)
        classification: bool = False,
):
    name = cfg.MODEL.MODEL_NAME
    N = cfg.MODEL.SSM_N
    d_model = cfg.MODEL.D_MODEL
    n_layers = cfg.MODEL.N_LAYERS
    dropout = cfg.MODEL.DROPOUT

    layer = get_layer_cls(name, N)
    return BatchStackedModel(
        layer=layer,
        d_output=d_output,
        d_model=d_model,
        seq_len=seq_len,
        n_layers=n_layers,
        dropout=dropout,
        decode=decode,
        classification=classification,
    )
