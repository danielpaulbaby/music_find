from pathlib import Path
import shutil
from collections import Counter

from sklearn.model_selection import train_test_split

# ======================================================
# Configuration
# ======================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# ======================================================
# Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = PROJECT_ROOT / "data" / "clips"

SOURCE_DIR = BASE_DIR / "original"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

# ======================================================
# Helper Functions
# ======================================================

def reset_directory(path: Path):
    """
    Deletes and recreates a directory.
    """

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def copy_files(file_list, label_list, destination):
    """
    Copies files into destination/class_name/
    """

    for file_path, label in zip(file_list, label_list):

        destination_folder = destination / label
        destination_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy2(file_path, destination_folder / file_path.name)


# ======================================================
# Validate Dataset
# ======================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Folder not found:\n{SOURCE_DIR}")

# ======================================================
# Read Dataset
# ======================================================

files = []
labels = []

print("\nScanning dataset...\n")

for class_folder in sorted(SOURCE_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    for audio_file in class_folder.iterdir():

        if audio_file.is_file():

            files.append(audio_file)
            labels.append(class_name)

print(f"Total clips found : {len(files)}")
print(f"Total classes     : {len(set(labels))}")

# ======================================================
# Create Train / Temp Split
# ======================================================

train_files, temp_files, train_labels, temp_labels = train_test_split(
    files,
    labels,
    train_size=TRAIN_RATIO,
    stratify=labels,
    random_state=RANDOM_STATE
)

# ======================================================
# Split Temp -> Validation + Test
# ======================================================

val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    train_size=val_ratio_adjusted,
    stratify=temp_labels,
    random_state=RANDOM_STATE
)

# ======================================================
# Reset Output Directories
# ======================================================

print("\nCreating dataset folders...\n")

reset_directory(TRAIN_DIR)
reset_directory(VAL_DIR)
reset_directory(TEST_DIR)

# ======================================================
# Copy Files
# ======================================================

print("Copying training clips...")
copy_files(train_files, train_labels, TRAIN_DIR)

print("Copying validation clips...")
copy_files(val_files, val_labels, VAL_DIR)

print("Copying testing clips...")
copy_files(test_files, test_labels, TEST_DIR)

# ======================================================
# Dataset Summary
# ======================================================

train_count = Counter(train_labels)
val_count = Counter(val_labels)
test_count = Counter(test_labels)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"{'Class':25}{'Train':>10}{'Val':>10}{'Test':>10}")

print("-" * 60)

classes = sorted(set(labels))

for class_name in classes:

    print(
        f"{class_name:25}"
        f"{train_count[class_name]:>10}"
        f"{val_count[class_name]:>10}"
        f"{test_count[class_name]:>10}"
    )

print("-" * 60)

print(
    f"{'TOTAL':25}"
    f"{len(train_files):>10}"
    f"{len(val_files):>10}"
    f"{len(test_files):>10}"
)

print("=" * 60)

# ======================================================
# Sanity Check
# ======================================================

train_set = set(train_files)
val_set = set(val_files)
test_set = set(test_files)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

print("\n✓ Dataset successfully split.")
print("✓ No duplicate clips across Train / Validation / Test.")