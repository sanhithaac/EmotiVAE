"""
modify_expression.py
--------------------
Shift the expression intensity of real faces by a given offset
using a trained EmotiVAE model.

Example
-------
    python modify_expression.py --expression_level=0.6 --img_size=50 --latent_dim=50
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from data_helpers import EmotionFaceDataset, compose_image_grid
from architectures import ConditionalVAE


def modify_expression(expression_level, IMG_SIZE, LATENT_DIM):
    """
    Perform a forward pass on training samples while shifting their
    expression score by *expression_level*, effectively making the faces
    more / less expressive.

    Parameters
    ----------
    expression_level : float
        Offset to add to the original expression score.
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
        shifted_scores = expression_level + 1.0 * batch["expression_score"]
        shifted_scores = shifted_scores.to(device)
        out = model(faces, shifted_scores)
        _, x_out = out["original"], out["reconstructed"]
        break

    result_imgs = []
    for tensor_img in x_out:
        arr = (255 * tensor_img).cpu().detach().numpy().astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        result_imgs.append(Image.fromarray(arr).convert("RGB"))

    grid = compose_image_grid(result_imgs, 4, 8, IMG_SIZE, IMG_SIZE)
    grid.save(f"outputs/modified/modified_{expression_level}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shift expression in face images")
    parser.add_argument(
        "--expression_level", help="Expression shift offset (default 0.6)",
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

    modify_expression(
        expression_level=cli.expression_level,
        IMG_SIZE=cli.img_size,
        LATENT_DIM=cli.latent_dim,
    )
