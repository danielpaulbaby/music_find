import os
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import MusicCNN
from dataloader import get_dataloader

# =========================
# CONFIG
# =========================
TRAIN_PATH = "data/spectrograms/train"
VAL_PATH = "data/spectrograms/val"

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
train_loader = get_dataloader(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = get_dataloader(
    VAL_PATH,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Number of classes
num_classes = len(train_loader.dataset.label_map)

print(f"Loaded {len(train_loader.dataset)} training samples")
print(f"Loaded {len(val_loader.dataset)} validation samples")
print(f"Number of classes: {num_classes}")

# =========================
# MODEL
# =========================
model = MusicCNN(num_classes=num_classes).to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# =========================
# CREATE SAVE DIRECTORY
# =========================
SAVE_DIR = "model/saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

best_val_accuracy = 0.0

# =========================
# TRAINING
# =========================
for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_accuracy = 100 * train_correct / train_total
    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    scheduler.step(val_accuracy)

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "music_model_best.pth")
        )

        print("✅ Best model updated!")

    # -------------------------
    # PRINT METRICS
    # -------------------------
    print("-" * 60)
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_accuracy:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_accuracy:.2f}%")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate : {current_lr:.6f}")
    print("-" * 60)

print(f"\nTraining Complete!")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print("Best model saved as:")
print("model/saved_model/music_model_best.pth")