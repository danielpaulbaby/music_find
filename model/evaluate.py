import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from cnn_model import MusicCNN
from dataloader import get_dataloader

# ==========================
# Configuration
# ==========================
MODEL_PATH = "model/saved_model/music_model_best.pth"
TEST_DIR = "data/spectrograms/test"

BATCH_SIZE = 32
NUM_CLASSES = 34

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Load Test Data
# ==========================
test_loader = get_dataloader(
    TEST_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# Load Model
# ==========================
model = MusicCNN(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = nn.CrossEntropyLoss()

# ==========================
# Evaluation
# ==========================
test_loss = 0
correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print("=" * 50)
print("TEST RESULTS")
print("=" * 50)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")
print("=" * 50)

# ==========================
# Classification Report
# ==========================
print("\nClassification Report\n")

target_names = [
    test_loader.dataset.idx_to_label[i]
    for i in range(len(test_loader.dataset.idx_to_label))
]

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=target_names
    )
)

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=test_loader.dataset.classes,
    yticklabels=test_loader.dataset.classes
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.show()