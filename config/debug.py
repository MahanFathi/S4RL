import ml_collections
from config.defaults import get_config as get_default_config

_C = ml_collections.ConfigDict()
_C.EXP_NAME = ""
_C.SEED = 0
_C.WANDB = True
_C.DEBUG = False
_C.MOCK_TPU = False


def get_config():
    _C = get_default_config()

    _C.DEBUG = True

    # mocks 8 tpu devices on cpu
    _C.MOCK_TPU = False

    _C.WANDB = False

    _C.MODEL.MODEL_NAME = "S4"
    _C.MODEL.SSM_N = 8
    _C.MODEL.D_MODEL = 12
    _C.MODEL.N_LAYERS = 2
    _C.MODEL.DROPOUT = 0.2

    return _C
