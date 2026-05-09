import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from video_dataset import VideoDataset
from video_model import DeepfakeVideoModel

# -----------------------------
# Dataset
# -----------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "extracted_frames")

dataset = VideoDataset(dataset_dir)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# -----------------------------
# Model
# -----------------------------

model = DeepfakeVideoModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# -----------------------------
# Training
# -----------------------------

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for sequences, labels in loader:

        labels = labels.unsqueeze(1)

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss: {total_loss}"
    )

# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "video_model.pth"
)

print("Training Complete")