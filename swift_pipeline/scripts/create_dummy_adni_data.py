"""
Create dummy ADNI data for testing the SwiFT pipeline
Generates synthetic tensors and JSON files that mimic real ADNI data structure
"""

import torch
import json
import os
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("CREATING DUMMY ADNI DATA FOR TESTING")
print("=" * 80)

# Configuration
NUM_SCANS = 10  # Total number of scans
SPATIAL_SHAPE = (91, 109, 91)  # Spatial dimensions (H, W, D)
TEMPORAL_LENGTH = 140  # Number of time points

# Output paths
OUTPUT_DIR = "../data_dummy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "all_4d_downsampled.pt")
LABELS_PATH = os.path.join(OUTPUT_DIR, "imageID_to_labels.json")
INFO_PATH = os.path.join(OUTPUT_DIR, "index_to_name.json")

print("\nConfiguration:")
print(f"  - Number of scans: {NUM_SCANS}")
print(f"  - Spatial shape: {SPATIAL_SHAPE}")
print(f"  - Temporal length: {TEMPORAL_LENGTH}")
print(f"  - Output directory: {OUTPUT_DIR}")

# ============================================================================
# 1. Create dummy 4D fMRI data tensor
# ============================================================================

print(f"\n{'=' * 80}")
print("STEP 1: Creating 4D fMRI tensor")
print(f"{'=' * 80}")

# Create realistic-looking fMRI data
print(
    f"Generating tensor of shape [{NUM_SCANS}, {SPATIAL_SHAPE[0]}, {SPATIAL_SHAPE[1]}, {SPATIAL_SHAPE[2]}, {TEMPORAL_LENGTH}]..."
)

# Initialize with random noise
data_4d = torch.randn(NUM_SCANS, *SPATIAL_SHAPE, TEMPORAL_LENGTH) * 0.5

# Add brain-like structure (higher signal in center)
center_h = SPATIAL_SHAPE[0] // 2
center_w = SPATIAL_SHAPE[1] // 2
center_d = SPATIAL_SHAPE[2] // 2

# Create a spherical brain-like mask
for h in range(SPATIAL_SHAPE[0]):
    for w in range(SPATIAL_SHAPE[1]):
        for d in range(SPATIAL_SHAPE[2]):
            distance = np.sqrt(
                ((h - center_h) / SPATIAL_SHAPE[0]) ** 2
                + ((w - center_w) / SPATIAL_SHAPE[1]) ** 2
                + ((d - center_d) / SPATIAL_SHAPE[2]) ** 2
            )
            if distance < 0.4:  # Brain region
                data_4d[:, h, w, d, :] += (
                    3.0 + torch.randn(NUM_SCANS, TEMPORAL_LENGTH) * 0.2
                )

# Add temporal correlations (make it more realistic)
for t in range(1, TEMPORAL_LENGTH):
    data_4d[:, :, :, :, t] = (
        0.7 * data_4d[:, :, :, :, t - 1] + 0.3 * data_4d[:, :, :, :, t]
    )

# Normalize
data_4d = (data_4d - data_4d.mean()) / (data_4d.std() + 1e-8)

print(f"✓ Created tensor: {data_4d.shape}")
print(f"  - Min value: {data_4d.min():.4f}")
print(f"  - Max value: {data_4d.max():.4f}")
print(f"  - Mean: {data_4d.mean():.4f}")
print(f"  - Std: {data_4d.std():.4f}")

# Save tensor
torch.save(data_4d, DATA_PATH)
print(f"✓ Saved tensor to: {DATA_PATH}")

# ============================================================================
# 2. Create index_to_name.json
# ============================================================================

print(f"\n{'=' * 80}")
print("STEP 2: Creating index_to_name.json")
print(f"{'=' * 80}")

index_to_name = {}

# Each scan has its own subject ID (simpler: 1 scan per subject)
np.random.seed(42)

# Generate metadata
base_date = datetime(2011, 1, 1)

for idx in range(NUM_SCANS):
    # Each scan has its own subject
    subject_id = f"012_S_{4000 + idx:04d}"

    # Generate scan date (spread over several years)
    days_offset = np.random.randint(0, 365 * 5)
    scan_date = base_date + timedelta(days=days_offset)
    date_str = scan_date.strftime("%d/%m/%Y")

    # Generate image ID
    image_id = f"I{300000 + idx:06d}"

    # Generate filename
    filename = f"dswau{subject_id}_{scan_date.strftime('%Y%m%d')}_Resting_State_fMRI_{80 + idx}_{image_id}.nii"

    index_to_name[str(idx)] = {
        "filename": filename,
        "subject_id": subject_id,
        "date": date_str,
        "image_id": image_id,
    }

print(f"✓ Created metadata for {NUM_SCANS} scans")
print(f"  Example entry:")
print(f"    {json.dumps(index_to_name['0'], indent=6)}")

# Save index_to_name
with open(INFO_PATH, "w") as f:
    json.dump(index_to_name, f, indent=4)
print(f"✓ Saved to: {INFO_PATH}")

# ============================================================================
# 3. Create imageID_to_labels.json with degradation tasks
# ============================================================================

print(f"\n{'=' * 80}")
print("STEP 3: Creating imageID_to_labels.json")
print(f"{'=' * 80}")

imageID_to_labels = {}

# Define realistic label distributions
np.random.seed(42)

for idx in range(NUM_SCANS):
    image_id = index_to_name[str(idx)]["image_id"]

    # Generate demographics
    age = np.random.uniform(55, 90)
    sex = np.random.choice(["M", "F"])

    # Generate cognitive scores
    mmse = np.random.randint(20, 31)  # MMSE score (20-30)
    gdscale = np.random.randint(0, 10)  # Depression scale (0-9)
    cdr = np.random.choice([0.0, 0.5, 1.0, 2.0])  # CDR global (0, 0.5, 1, 2)
    faq = np.random.randint(0, 20)  # FAQ score (0-20)

    # Generate CDR sub-scores
    cdmemory = np.random.choice([0.0, 0.5, 1.0, 2.0])
    cdorient = np.random.choice([0.0, 0.5, 1.0])
    cdjudge = np.random.choice([0.0, 0.5, 1.0])
    cdcommun = np.random.choice([0.0, 0.5, 1.0])
    cdhome = np.random.choice([0.0, 0.5, 1.0])
    cdcare = np.random.choice([0.0, 0.5, 1.0])
    cdrsb = cdmemory + cdorient + cdjudge + cdcommun + cdhome + cdcare

    # Generate degradation labels (binary)
    # Higher CDR and FAQ scores correlate with higher degradation risk
    degradation_prob = min(0.8, (cdr + faq / 20) / 3)

    degradation_1year = int(np.random.random() < degradation_prob * 0.3)
    degradation_2years = int(np.random.random() < degradation_prob * 0.5)
    degradation_3years = int(np.random.random() < degradation_prob * 0.7)

    # Create categorical versions
    sex_binary = 1 if sex == "M" else 0
    age_category = int(age // 10) - 5  # 0: 50s, 1: 60s, 2: 70s, 3: 80s+
    mmse_binary = 1 if mmse < 24 else 0
    gdscale_category = min(3, gdscale // 3)
    cdr_category = int(cdr * 2)  # 0, 1, 2, 4
    cdr_binary = 1 if cdr >= 0.5 else 0
    faq_binary = 1 if faq >= 5 else 0

    # Randomly add some NaN values (to test NaN handling)
    add_nan = np.random.random() < 0.1  # 10% chance of NaN

    imageID_to_labels[image_id] = {
        "Sex": sex,
        "Age": float(age),
        "MMSE Total Score": float(mmse) if not add_nan else float("nan"),
        "GDSCALE Total Score": float(gdscale),
        "Global CDR": float(cdr),
        "FAQ Total Score": float(faq),
        "NPI-Q Total Score": float(np.random.randint(0, 15))
        if np.random.random() > 0.2
        else float("nan"),
        "Sex_Binary": sex_binary,
        "Age_Category": age_category,
        "MMSE_Binary": mmse_binary,
        "GDSCALE_Category": gdscale_category,
        "CDR_Category": cdr_category,
        "CDR_Binary": cdr_binary,
        "FAQ_Binary": faq_binary,
        "CDSOURCE": float(np.random.choice([0.0, 1.0]))
        if np.random.random() > 0.3
        else float("nan"),
        "CDVERSION": float(np.random.choice([0.0, 1.0]))
        if np.random.random() > 0.3
        else float("nan"),
        "SPID": float("nan"),
        "CDMEMORY": float(cdmemory),
        "CDORIENT": float(cdorient),
        "CDJUDGE": float(cdjudge),
        "CDCOMMUN": float(cdcommun),
        "CDHOME": float(cdhome),
        "CDCARE": float(cdcare),
        "CDGLOBAL": float(cdr),
        "CDRSB": float(cdrsb),
        "degradation_binary_1year": degradation_1year,
        "degradation_binary_2years": degradation_2years,
        "degradation_binary_3years": degradation_3years,
    }

# Count degradation labels
deg_1y = sum(
    1 for v in imageID_to_labels.values() if v["degradation_binary_1year"] == 1
)
deg_2y = sum(
    1 for v in imageID_to_labels.values() if v["degradation_binary_2years"] == 1
)
deg_3y = sum(
    1 for v in imageID_to_labels.values() if v["degradation_binary_3years"] == 1
)

print(f"✓ Created labels for {NUM_SCANS} scans")
print(
    f"  - Degradation 1-year:  {deg_1y}/{NUM_SCANS} positive ({100 * deg_1y / NUM_SCANS:.1f}%)"
)
print(
    f"  - Degradation 2-years: {deg_2y}/{NUM_SCANS} positive ({100 * deg_2y / NUM_SCANS:.1f}%)"
)
print(
    f"  - Degradation 3-years: {deg_3y}/{NUM_SCANS} positive ({100 * deg_3y / NUM_SCANS:.1f}%)"
)
print(f"  - Example entry:")
example_id = list(imageID_to_labels.keys())[0]
print(f"    {example_id}: {{")
for k, v in list(imageID_to_labels[example_id].items())[:5]:
    print(f"      {k}: {v}")
print(f"      ...")
print(f"    }}")

# Save labels
with open(LABELS_PATH, "w") as f:
    json.dump(imageID_to_labels, f, indent=4)
print(f"✓ Saved to: {LABELS_PATH}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'=' * 80}")
print("DUMMY DATA CREATION COMPLETE")
print(f"{'=' * 80}")

print(f"\n📁 Generated Files:")
print(f"  1. {DATA_PATH}")
print(f"     - Shape: {data_4d.shape}")
print(f"     - Size: {os.path.getsize(DATA_PATH) / (1024**2):.2f} MB")
print(f"\n  2. {INFO_PATH}")
print(f"     - Entries: {len(index_to_name)}")
print(f"     - Size: {os.path.getsize(INFO_PATH) / 1024:.2f} KB")
print(f"\n  3. {LABELS_PATH}")
print(f"     - Entries: {len(imageID_to_labels)}")
print(f"     - Size: {os.path.getsize(LABELS_PATH) / 1024:.2f} KB")

print(f"\n✅ Ready to use with validate_adni_pipeline.ipynb!")
print(f"   Update the notebook configuration to point to: {OUTPUT_DIR}")
print(f"{'=' * 80}\n")
