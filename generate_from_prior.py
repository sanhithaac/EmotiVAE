"""
generate_from_prior.py
----------------------
Sample the latent space from the prior distribution and decode
synthetic faces at a given expression level using a trained EmotiVAE.

Example
-------
    python generate_from_prior.py --expression_level=0.6 --img_size=50 --latent_dim=50
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from data_helpers import EmotionFaceDataset, compose_image_grid
from architectures import ConditionalVAE


def sample_from_prior(expression_level, IMG_SIZE, LATENT_DIM):
    """
    Draw random samples from the learned prior, condition them with
    *expression_level*, and decode into synthetic face images.

    Parameters
    ----------
    expression_level : float
        Desired expression intensity for the generated faces.
    IMG_SIZE : int
        Pixel size the model was trained with.
    LATENT_DIM : int
        Latent dimensionality the model was trained with.
    """
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
        faces = batch["face"].to(device)
        x_synth, _ = model.draw_from_prior(SAMPLES_PER_BATCH, expression_level)
        break

    synth_imgs = []
    for tensor_img in x_synth:
        arr = (255 * tensor_img).cpu().detach().numpy().astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        synth_imgs.append(Image.fromarray(arr).convert("RGB"))

    grid = compose_image_grid(synth_imgs, 4, 8, IMG_SIZE, IMG_SIZE)
    grid.save(f"outputs/sampled/sampled_{expression_level}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate faces from the prior")
    parser.add_argument(
        "--expression_level", help="Expression intensity (default 0.6)",
        default=0.6, type=float,
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

    sample_from_prior(
        expression_level=cli.expression_level,
        IMG_SIZE=cli.img_size,
        LATENT_DIM=cli.latent_dim,
    )
