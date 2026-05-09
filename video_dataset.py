import os
import cv2
import torch

from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):

    def __init__(self, root_dir, sequence_length=16):

        self.samples = []
        self.labels = []

        self.sequence_length = sequence_length

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # -----------------------------
        # Real videos
        # -----------------------------

        real_path = os.path.join(root_dir, "real")

        for video_folder in os.listdir(real_path):

            full_path = os.path.join(real_path, video_folder)

            self.samples.append(full_path)

            self.labels.append(0)

        # -----------------------------
        # Fake videos
        # -----------------------------

        fake_path = os.path.join(root_dir, "fake")

        for video_folder in os.listdir(fake_path):

            full_path = os.path.join(fake_path, video_folder)

            self.samples.append(full_path)

            self.labels.append(1)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        video_folder = self.samples[index]

        frames = sorted(os.listdir(video_folder))

        sequence = []

        # -----------------------------
        # Load first 16 frames
        # -----------------------------

        for frame_name in frames[:self.sequence_length]:

            frame_path = os.path.join(
                video_folder,
                frame_name
            )

            img = cv2.imread(frame_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)

            sequence.append(img)

        # -----------------------------
        # Handle short videos
        # -----------------------------

        while len(sequence) < self.sequence_length:

            sequence.append(sequence[-1])

        sequence = torch.stack(sequence)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return sequence, label