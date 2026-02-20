"""
data_helpers.py
---------------
Dataset class and image utility functions for EmotiVAE.
Handles loading the UTKFace expression dataset and building image grids.
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class EmotionFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for the UTK-Face expression subset.

    Each sample consists of a face image and a continuous expression-intensity
    score (ranging roughly from -1 to +1).

    Parameters
    ----------
    annotations_path : str
        Path to the CSV / TSV file that maps image filenames to expression scores.
    image_dir : str
        Root folder containing the face images.
    img_transform : callable, optional
        Torchvision transforms (or any callable) applied to every image.
    """

    def __init__(self, annotations_path, image_dir="datasets/images", img_transform=None):
        dataframe = pd.read_csv(annotations_path)
        self.filenames = dataframe["stimulus_filename"].tolist()
        self.scores = dataframe["avg_encode"].tolist()
        self.image_dir = image_dir
        self.img_transform = img_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Fetches the image at *index*, converts it to RGB (to handle
        mixed formats such as RGBA or Grayscale), and returns it together
        with the expression score.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.image_dir, self.filenames[index])
        face_img = Image.open(img_path).convert("RGB")
        score = self.scores[index]

        sample = {"face": face_img, "expression_score": score}

        if self.img_transform:
            sample["face"] = self.img_transform(sample["face"])

        return sample


def compose_image_grid(images, n_rows, n_cols, cell_h=50, cell_w=50):
    """
    Arranges a flat list of PIL Images into a grid.

    Parameters
    ----------
    images : list[PIL.Image]
        Exactly ``n_rows * n_cols`` images.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    cell_h : int
        Height (px) of every cell.
    cell_w : int
        Width (px) of every cell.

    Returns
    -------
    PIL.Image
        The composited grid image.
    """
    assert len(images) == n_rows * n_cols, (
        f"Expected {n_rows * n_cols} images, got {len(images)}"
    )

    canvas = Image.new("RGB", size=(n_cols * cell_w, n_rows * cell_h))
    for idx, img in enumerate(images):
        col = idx % n_cols
        row = idx // n_cols
        canvas.paste(img, box=(col * cell_w + 1, row * cell_h + 1))
    return canvas
