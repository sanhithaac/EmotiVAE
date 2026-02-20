"""
explore_pca_axis.py
-------------------
Like explore_latent_axis but operates in PCA-transformed latent space.
Sweeps a principal-component direction to visualise disentangled factors.

Example
-------
    python explore_pca_axis.py --axis=0 --img_size=50 --latent_dim=50
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image

from data_helpers import EmotionFaceDataset, compose_image_grid
from architectures import ConditionalVAE


def explore_pca_axis(axis_idx, IMG_SIZE, LATENT_DIM):
    """
    Perform PCA on the stored latent representations, sweep the
    *axis_idx*-th principal component, inverse-transform, and decode.

    Parameters
    ----------
    axis_idx : int
        Index of the principal component to explore.
    IMG_SIZE : int
        Pixel size the model was trained with.
    LATENT_DIM : int
        Latent dimensionality the model was trained with.
    """
    assert 0 <= axis_idx <= LATENT_DIM, (
        f"axis_idx must satisfy 0 <= {axis_idx} <= {LATENT_DIM}"
    )
    SAMPLES_PER_BATCH = 32

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    dataset = EmotionFaceDataset(
        annotations_path="datasets/smiles_dataset.txt",
        image_dir="datasets/images/",
        img_transform=eval_transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=SAMPLES_PER_BATCH)

    for batch in loader:
        input_tensor_shape = batch["face"].shape
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(img_shape=input_tensor_shape, latent_dim=LATENT_DIM).to(device)

    ckpt_path = "checkpoints/model.pt"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    else:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu"))["model_state_dict"]
        )

    model.eval()
    for batch in loader:
        _, z_cond = model.draw_from_prior(SAMPLES_PER_BATCH, 0.0)
        break

    # load pre-computed latent representations for PCA
    latent_data = np.genfromtxt("outputs/latent.txt")
    pca = PCA()
    pca.fit_transform(latent_data)

    decoded_tensors = []
    for val in np.linspace(-1.2, 1.2, SAMPLES_PER_BATCH):
        z_probe = 0.0 * z_cond
        z_probe[0][axis_idx] = val
        z_probe[0][:-1] = torch.from_numpy(
            pca.inverse_transform(z_probe[0][:-1].numpy())
        )
        decoded_tensors.append(model.reconstruct(z_probe)[0])

    result_imgs = []
    for tensor_img in decoded_tensors:
        arr = (255 * tensor_img).cpu().detach().numpy().astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        result_imgs.append(Image.fromarray(arr).convert("RGB"))

    grid = compose_image_grid(result_imgs, 4, 8, IMG_SIZE, IMG_SIZE)
    grid.save(f"outputs/axis/pca_axis_{axis_idx}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore a PCA latent axis")
    parser.add_argument(
        "--axis", help="PCA component index (default 0)", default=0, type=int
    )
    parser.add_argument(
        "--img_size", help="Resized dimension in pixels (default 50)",
        default=50, type=int,
    )
    parser.add_argument(
        "--latent_dim", help="Latent dimensionality (default 50)",
        default=50, type=int,
    )
    cli = parser.parse_args()

    explore_pca_axis(axis_idx=cli.axis, IMG_SIZE=cli.img_size, LATENT_DIM=cli.latent_dim)
