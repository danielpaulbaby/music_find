import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to spectrogram folder
        """
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.label_map = {}

        # Get sorted song folders (important for consistency)
        songs = sorted(os.listdir(data_dir))

        for label, song in enumerate(songs):
            song_path = os.path.join(data_dir, song)

            # Skip non-directories (safety)
            if not os.path.isdir(song_path):
                continue

            self.label_map[song] = label
            self.idx_to_label = {v: k for k, v in self.label_map.items()}

            for file in os.listdir(song_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(song_path, file)
                    self.samples.append(file_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.samples)} samples from {len(self.label_map)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load spectrogram
        spec = np.load(self.samples[idx])

        # Normalize (VERY important)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)

        # Convert to tensor
        spec = torch.tensor(spec, dtype=torch.float32)

        # Add channel dimension → (1, H, W)
        spec = spec.unsqueeze(0)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spec, label


# 🔥 Optional helper function (cleaner usage)
def get_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = SpectrogramDataset(data_dir)

    return torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=0,
    pin_memory=False
)