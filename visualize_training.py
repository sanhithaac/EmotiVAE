"""
visualize_training.py
---------------------
Post-training analysis: PCA on latent representations,
latent variance heatmaps, and training-curve summaries.

Example
-------
    python visualize_training.py --img_size=50 --latent_dim=50
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image

from data_helpers import EmotionFaceDataset, compose_image_grid
from architectures import ConditionalVAE


def compute_latent_subspace(IMG_SIZE, LATENT_DIM):
    """
    Forward-pass the full training set through the trained model,
    collect all latent vectors, run PCA, and save the explained-
    variance plot.
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

    latent_buffer = []
    model.eval()
    for batch in loader:
        faces = batch["face"].to(device)
        scores = batch["expression_score"].to(device)
        out = model(faces, scores)
        for z_i in out["z_sample"]:
            latent_buffer.append(z_i.cpu().detach().numpy())

    latent_array = np.array(latent_buffer)

    pca = PCA()
    pca.fit_transform(latent_array)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_), "*-")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.savefig("outputs/plots/latent_pca.png")
    plt.close()


def plot_latent_variance():
    """Parse the per-epoch latent variance log and render heatmaps."""
    with open("outputs/latent_variance.txt", "r") as fh:
        raw_lines = fh.readlines()

    parsed_epochs, current_block = [], []
    for line in raw_lines:
        if "[" in line:
            if len(current_block) > 1:
                current_block = [
                    tok for chunk in current_block for tok in chunk.split()
                ]
                parsed_epochs.append(current_block)
            current_block = []
            current_block.append(line.replace("[", ""))
        else:
            current_block.append(line.replace("]", ""))

    latent_matrix = np.array(parsed_epochs, dtype=float)
    np.savetxt("outputs/latent.txt", latent_matrix)

    # first 50 epochs
    var_data = latent_matrix[:50].T
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    heatmap = ax.imshow(var_data, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Dimension")
    plt.colorbar(heatmap, ax=ax)
    plt.savefig("outputs/plots/latent_var_first_50.png")
    plt.close()

    # last 50 epochs
    var_data = latent_matrix[-50:-1].T
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    heatmap = ax.imshow(var_data, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Dimension")
    plt.colorbar(heatmap, ax=ax)
    plt.savefig("outputs/plots/latent_var_last_50.png")
    plt.close()

    # sampled every 20 epochs
    var_data = latent_matrix[::20].T
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    heatmap = ax.imshow(var_data, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Dimension")
    plt.colorbar(heatmap, ax=ax)
    plt.savefig("outputs/plots/latent_var.png")
    plt.close()


def plot_training_curves():
    """Read the training log and produce loss / LR plots."""
    log_data = np.genfromtxt("outputs/training_log.txt")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].set_title("Total Loss (ELBO)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(log_data[:, 0], log_data[:, 1])

    axes[1, 0].set_title("MSE Loss (Reconstruction)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].plot(log_data[:, 0], log_data[:, 2], "green")

    axes[1, 1].set_title("KLD Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("KLD")
    axes[1, 1].plot(log_data[:, 0], log_data[:, 3], "orange")

    axes[0, 1].set_title("Learning Rate")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("LR")
    axes[0, 1].plot(log_data[:, 0], log_data[:, 4], "red")

    plt.tight_layout()
    plt.savefig("outputs/plots/summary_training.png")
    plt.close()

    # standalone loss plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_title("Total Loss (ELBO)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(log_data[:, 0], log_data[:, 1])
    plt.savefig("outputs/plots/training.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-training visualisations")
    parser.add_argument(
        "--img_size", help="Resized dimension (pixels, default 50)",
        default=50, type=int,
    )
    parser.add_argument(
        "--latent_dim", help="Latent dimensionality (default 50)",
        default=50, type=int,
    )
    cli = parser.parse_args()

    compute_latent_subspace(IMG_SIZE=cli.img_size, LATENT_DIM=cli.latent_dim)
    plot_latent_variance()
    plot_training_curves()
