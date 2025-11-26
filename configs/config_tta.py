import torch

# --- Paths and Directories ---
BASE_DATA_PATH = "path/to/your/data"
# Path to the pretrained model (encoder/decoder)
PRETRAINED_CHECKPOINT_PATH = "results/pretraining_runs/your_pretrain_run_id/model.pt"
# Path to the fine-tuned prediction head
FINETUNED_HEAD_PATH = "results/finetuning_runs/your_finetune_run_id/model.pt"

# --- Data Settings ---
TEST_SPLIT = 0.2  # Should match previous configs
SEED = 44
WINDOW_SIZE = 10

# --- TTA Task Configuration ---
# Must match the task the head was fine-tuned on
TTA_TASK = "degradation_binary_1year"
TASKS_TYPES = {TTA_TASK: "binary"}
CHOSEN_LABELS = [TTA_TASK]

# --- TTA Hyperparameters ---

TTA_STEPS = 50
TTA_LR = 5e-4 * 0.1
# Use a batch size of 1 for sample-by-sample adaptation
BATCH_SIZE = 1

# --- Runtime and Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = 1
