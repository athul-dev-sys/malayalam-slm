"""uint16-memmap token dataset with random-window batching (documents joined by EOS)."""
import numpy as np
import torch


class TokenMemmap:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def get_batch(self, block_size: int, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
