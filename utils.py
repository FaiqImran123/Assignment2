import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = ("png", "jpg", "jpeg", "bmp", "webp")


def find_image_dir(data_path, data_class="cat"):
    """Return data_path/data_class if it exists, otherwise data_path."""
    cand = os.path.join(data_path, data_class)
    return cand if os.path.isdir(cand) else data_path


class AFHQDataset(Dataset):
    """Loads all images under a folder (recursively) and scales them to [-1, 1]."""

    def __init__(self, folder, image_size=32):
        self.files = []
        for ext in IMG_EXTS:
            self.files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
            self.files += glob.glob(os.path.join(folder, "**", f"*.{ext.upper()}"), recursive=True)
        self.files = sorted(set(self.files))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {folder}. Check data_path.")
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img)


def cycle(dl):
    """Infinite dataloader iterator."""
    while True:
        for batch in dl:
            yield batch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
