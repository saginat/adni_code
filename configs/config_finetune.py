from datetime import datetime
import torch

# --- Paths and Directories ---
BASE_DATA_PATH = "path/to/your/data"
BASE_RESULTS_PATH = "results/"
# Path to the best checkpoint from the pretraining run
PRETRAINED_CHECKPOINT_PATH = "results/pretraining_runs/your_pretrain_run_id/model.pt"
FINETUNE_MODEL_DIR = "results/finetuning_runs/"


# --- Data Settings ---
# These should match the pretraining config to ensure data consistency
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 44
WINDOW_SIZE = 10

# --- Fine-tuning Task Configuration ---
# Example: Predicting cognitive decline at 1-year horizon
FINETUNE_TASK = "degradation_binary_1year"
TASKS_TYPES = {
    FINETUNE_TASK: "binary",
}
OUTPUT_DIMS = {FINETUNE_TASK: 1}
CHOSEN_LABELS = [FINETUNE_TASK]
TASK_WEIGHTS = {FINETUNE_TASK: 1.0}

# --- Fine-tuning Head & Training Hyperparameters ---
# The input dimension must match the flattened bottleneck dimension from the pretrained model
# (MERGE_PATCHES * EMBEDDING_DIM from pretrain config = 10 * 128 = 1280)
HEAD_INPUT_DIM = 1280
HEAD_P_DROPOUT = 0.2
LR = 3e-6  #
NUM_EPOCHS = 2
BATCH_SIZE = 4
OPTIMIZER_WEIGHT_DECAY = 1e-5

# --- Runtime and Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1
START_DATE = datetime.now().strftime("%d/%m/%Y")

# --- Checkpointing ---
CHECKPOINT_PERCENTAGE = 0.0  # Only save at the end
BEST_METRIC_NAME = "f1"  # Monitor F1-score for best model


# # --- Task and Loss Configuration ---

# TASKS = [
#     "Sex_Binary",
#     "MMSE_Binary",
#     "CDR_Binary",
#     "FAQ_Binary",
#     "Age_Category",
#     "GDSCALE_Category",
#     "CDR_Category",
#     "CDMEMORY",
#     "CDRSB",
#     "degradation_binary_1year",
#     "degradation_binary_2years",
#     "degradation_binary_3years",
# ]

# # Weights for each task's loss
# TASK_WEIGHTS = {
#     "Sex_Binary": 1,
#     "MMSE_Binary": 1,
#     "CDR_Binary": 1,
#     "FAQ_Binary": 1,
#     "Age_Category": 1,
#     "GDSCALE_Category": 1,
#     "CDR_Category": 1,
#     "CDMEMORY": 1,
#     "CDRSB": 1,
#     "Reconstruction": 1,
# }

# # Mapping of task names to their type ('binary', 'categorical', 'regression')
# TASKS_TYPES = {
#     "Sex_Binary": "binary",
#     "MMSE_Binary": "binary",
#     "CDR_Binary": "binary",
#     "FAQ_Binary": "binary",
#     "Age_Category": "categorical",
#     "GDSCALE_Category": "categorical",
#     "CDR_Category": "categorical",
#     "CDMEMORY": "categorical",
#     "CDRSB": "categorical",
#     "Reconstruction": "regression",
#     "degradation_binary_1year": "binary",
#     "degradation_binary_2years": "binary",
#     "degradation_binary_3years": "binary",
# }


# CATEGORY_TO_DIM = {
#     "Sex_Binary": 1,
#     "MMSE_Binary": 1,
#     "CDR_Binary": 1,
#     "FAQ_Binary": 1,
#     "Age_Category": 4,
#     "GDSCALE_Category": 4,
#     "CDR_Category": 4,
#     "CDMEMORY": 4,
#     "CDRSB": 19,
#     "degradation_binary_1year": 1,
#     "degradation_binary_2years": 1,
#     "degradation_binary_3years": 1,
# }
