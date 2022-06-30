import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()
_C.EXP_NAME = ""
_C.SEED = 0
_C.WANDB = True
_C.DEBUG = False
_C.MOCK_TPU = False

# ---------------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------------- #
_C.ENV = ml_collections.ConfigDict()
_C.ENV.ENV_NAME = "walker2d"

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATA = ml_collections.ConfigDict()
_C.DATA.DS_NAME = "d4rl/medium"
#   d4rl from [medium, medium-replay, medium-expert]

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = ml_collections.ConfigDict()
_C.MODEL.MODEL_NAME = "S4"
_C.MODEL.SSM_N = 32
_C.MODEL.D_MODEL = 64
_C.MODEL.N_LAYERS = 16
_C.MODEL.DROPOUT = 0.2
_C.MODEL.SEQ_LEN = 1000

# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #
_C.TRAIN = ml_collections.ConfigDict()
_C.TRAIN.EPOCHS = 10000
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.LR = 1e-3
_C.TRAIN.LR_SCHEDULE = True

# ---------------------------------------------------------------------------- #
# DEFAULT CONFIG
# ---------------------------------------------------------------------------- #
def get_config():
    return _C
