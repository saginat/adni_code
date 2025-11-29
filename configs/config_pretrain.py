from datetime import datetime
import torch

# --- Paths and Directories ---
BASE_DATA_PATH = "/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data/"
BASE_RESULTS_PATH = "/sci/nosnap/arieljaffe/sagi.nathan/results/"
PRETRAINED_MODEL_DIR = "/sci/nosnap/arieljaffe/sagi.nathan/tracker/pretraining_runs/"
ATLAS = "schaefer200"

# --- Data Settings ---
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 44
WINDOW_SIZE = 10
REMOVE_TOP_K_STD = 7

# --- Model Architecture ---
PATCH_SIZE = (6, 6, 6)
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
NUM_CLS_TOKENS = 0
P_DROPOUT = 0.1
REDUCED_PATCHES_FACTOR_PERCENT = 0.1
REDUCE_TIME_FACTOR_PERCENT = 0.2
MERGE_PATCHES = 10
USE_PATCH_MERGER = True
CUSTOM_RECON_BOOL = True

# --- Training Configuration ---
TASKS = []
TASK_WEIGHTS = {"Reconstruction": 1.0}
TASKS_TYPES = {"Reconstruction": "regression"}
CHOSEN_LABELS = ["Reconstruction"]

LR = 5e-4
NUM_EPOCHS = 55
BATCH_SIZE = 8
OPTIMIZER_WEIGHT_DECAY = 0.0005
MAX_NORM = 1.0
WARMUP_STEPS_PERCENT = 0.1


# --- Runtime and Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1  # 0: silent, 1: epoch tqdm, 2: epoch and batch tqdm
START_DATE = datetime.now().strftime("%d/%m/%Y")
TRACK_GRAD = True

# --- Checkpointing and Plotting ---
CHECKPOINT_PERCENTAGE = (
    0.0  # Percentage of epochs to save a checkpoint, 0 saves only at the end
)
CHECKPOINT_TASK = "Reconstruction"  # Task to monitor for best model saving
BEST_METRIC_NAME = "loss"  # Metric to monitor ('loss', 'accuracy', 'f1', etc.)
PLOT_PERCENTAGE = 0.25  # Percentage of epochs to generate and save plots
MAX_ROWS_FOR_PLOT = 2
