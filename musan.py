import numpy as np
import os

widths = []

for root, dirs, files in os.walk("data/spectrograms/noisy"):
    for file in files:
        if file.endswith(".npy"):
            spec = np.load(os.path.join(root, file))
            widths.append(spec.shape[1])

print(min(widths), max(widths))