"""
train_model.py
--------------
Main training script for the EmotiVAE Conditional VAE.
Trains on the UTKFace expression dataset and periodically saves
reconstructions and model checkpoints.

Example
-------
    python train_model.py --total_epochs=1000 --lr=0.00025 --kld_weight=0.5 \
                          --batch_count=8 --latent_dim=20 --img_size=50
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from data_helpers import EmotionFaceDataset, compose_image_grid
from monitoring import LatentVisualizer
from architectures import ConditionalVAE


def train(LATENT_DIM, TOTAL_EPOCHS, LR, BATCH_COUNT, IMG_SIZE, KLD_WEIGHT):
    """
    Full training loop for the Conditional VAE.

    Parameters
    ----------
    LATENT_DIM : int   – dimensionality of the latent space
    TOTAL_EPOCHS : int – number of training epochs
    LR : float         – initial learning rate
    BATCH_COUNT : int  – mini-batch size
    IMG_SIZE : int     – resize target (pixels, square)
    KLD_WEIGHT : float – weight multiplier for the KL-divergence term
    """
    EVAL_INTERVAL = 5

    # ---------- data preparation ----------
    augment_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = EmotionFaceDataset(
        annotations_path="datasets/smiles_dataset.txt",
        image_dir="datasets/images/",
        img_transform=augment_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_COUNT, shuffle=True
    )

    eval_dataset = EmotionFaceDataset(
        annotations_path="datasets/smiles_dataset.txt",
        image_dir="datasets/images/",
        img_transform=eval_transform,
    )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)

    # infer the input tensor shape from a single batch
    for batch in train_loader:
        input_tensor_shape = batch["face"].shape
        break

    # ---------- model / optimizer / scheduler ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalVAE(img_shape=input_tensor_shape, latent_dim=LATENT_DIM).to(device)
    latent_monitor = LatentVisualizer(
        n_dims=LATENT_DIM, output_path="outputs/latent_variance.txt"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # cosine annealing instead of step-wise decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )

    # ---------- training loop ----------
    model.train()
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            faces = batch["face"].to(device)
            scores = batch["expression_score"].to(device)

            out = model(faces, scores)
            p_z = out["p_z"]
            q_z = out["q_z"]
            z_sample = out["z_sample"]
            x_in = out["original"]
            x_out = out["reconstructed"]

            latent_monitor(z_sample.detach().cpu().numpy())

            # reconstruction loss (MSE) + KL divergence
            mse_loss = torch.mean((x_out - x_in) ** 2)
            kl_loss = -KLD_WEIGHT * torch.mean(
                torch.mean(
                    1 + torch.log(q_z.std ** 2) - q_z.mean ** 2 - torch.exp(q_z.std ** 2)
                )
            )
            total_loss = mse_loss + kl_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.detach().cpu().item()

        latent_monitor.finish_epoch()
        scheduler.step()

        # ---------- evaluation / visualisation ----------
        if (epoch % EVAL_INTERVAL == 0) or (epoch == 1):
            model.eval()

            for batch in eval_loader:
                optimizer.zero_grad()
                faces = batch["face"].to(device)
                scores = batch["expression_score"].to(device)
                out = model(faces, scores)
                x_in, x_out = out["original"], out["reconstructed"]
                break

            recon_imgs = []
            for tensor_img in x_out:
                arr = (255 * tensor_img).cpu().detach().numpy().astype(np.uint8)
                arr = np.transpose(arr, (1, 2, 0))
                recon_imgs.append(Image.fromarray(arr).convert("RGB"))

            grid = compose_image_grid(recon_imgs, 4, 8, IMG_SIZE, IMG_SIZE)
            grid.save(f"outputs/iterations/iter_{epoch}.png")

            if epoch == 1:
                orig_imgs = []
                for tensor_img in x_in:
                    arr = (255 * tensor_img).cpu().detach().numpy().astype(np.uint8)
                    arr = np.transpose(arr, (1, 2, 0))
                    orig_imgs.append(Image.fromarray(arr).convert("RGB"))
                grid = compose_image_grid(orig_imgs, 4, 8, IMG_SIZE, IMG_SIZE)
                grid.save("outputs/iterations/original.png")

            model.train()

        # ---------- checkpoint ----------
        if epoch % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                "checkpoints/model.pt",
            )

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch:{epoch}, loss:{epoch_loss:.4f}, "
            f"mse:{mse_loss:.4f}, kld:{kl_loss:.4f}, lr:{current_lr:.6f}"
        )
        with open("outputs/training_log.txt", "a") as log_fh:
            print(epoch, epoch_loss, mse_loss.item(), kl_loss.item(), current_lr, file=log_fh)

    # final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        },
        "checkpoints/model.pt",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the EmotiVAE model")
    parser.add_argument(
        "--total_epochs", help="Number of training epochs (default 100)",
        default=100, type=int,
    )
    parser.add_argument(
        "--lr", help="Initial learning rate (default 0.00025)",
        default=0.00025, type=float,
    )
    parser.add_argument(
        "--kld_weight", help="KL-divergence weight (default 0.5)",
        default=0.5, type=float,
    )
    parser.add_argument(
        "--batch_count", help="Mini-batch size (default 8)",
        default=8, type=int,
    )
    parser.add_argument(
        "--latent_dim", help="Latent space dimensionality (default 100)",
        default=100, type=int,
    )
    parser.add_argument(
        "--img_size", help="Resize target in pixels (default 20)",
        default=20, type=int,
    )
    cli = parser.parse_args()

    train(
        LATENT_DIM=cli.latent_dim,
        TOTAL_EPOCHS=cli.total_epochs,
        LR=cli.lr,
        BATCH_COUNT=cli.batch_count,
        IMG_SIZE=cli.img_size,
        KLD_WEIGHT=cli.kld_weight,
    )
