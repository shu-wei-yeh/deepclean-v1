# Dataset default
DEFAULT_SAMPLE_RATE = 16384
DEFAULT_TRAIN_FRAC = 0.9

# Training default
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_MAX_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3

# Early stopping default
DEFAULT_EARLY_STOPPING = True
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
# Do not allow early stopping before this many epochs have completed.
# This gives the StepLR scheduler (step_size=10) a chance to reduce the LR.
DEFAULT_EARLY_STOPPING_MIN_EPOCHS = 12

# Timeseries default
DEFAULT_TRAIN_KERNEL = 8.
DEFAULT_TRAIN_STRIDE = 0.25
DEFAULT_CLEAN_KERNEL = 8.
DEFAULT_CLEAN_STRIDE = 4.
DEFAULT_PAD_MODE = 'median'
DEFAULT_WINDOW = 'hanning'

# Preprocess default
DEFAULT_FLOW = 20
DEFAULT_FHIGH = 8192
DEFAULT_FORDER = 8

# Loss default
# Historical DeepClean baseline used for this study.
# COH/TF are optional additive regularizers; the weights do not need
# to sum to 1.
DEFAULT_FFT_LENGTH = 2
DEFAULT_OVERLAP = None
DEFAULT_PSD_WEIGHT = 1.0
DEFAULT_MSE_WEIGHT = 0.0
DEFAULT_COH_WEIGHT = 0.0
DEFAULT_TF_WEIGHT = 0.0

# Device
DEFAULT_DEVICE = 'cuda'
