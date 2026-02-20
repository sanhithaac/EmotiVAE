"""
download_dataset.py
-------------------
Downloads the UTKFace Aligned & Cropped dataset from Google Drive,
extracts it, and keeps only the 186 images referenced in smiles_dataset.txt.

Run once:
    python download_dataset.py
"""

import os
import zipfile
import shutil
import pandas as pd


def download_and_prepare():
    csv_path = "datasets/smiles_dataset.txt"
    image_dir = "datasets/images"
    temp_zip = "datasets/utk_crop.zip"
    temp_extract = "datasets/_temp_extract"

    # Read required filenames from CSV
    df = pd.read_csv(csv_path)
    required_files = set(df["stimulus_filename"].tolist())
    print(f"Need {len(required_files)} images from UTKFace")

    # Download from Google Drive using gdown
    # UTKFace Aligned & Cropped (crop_part1) file ID
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
        import gdown

    # The UTKFace crop_part1.tar.gz Google Drive file ID
    gdrive_url = "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"

    print("Downloading UTKFace Aligned & Cropped dataset (~107MB)...")
    print("If this fails, download manually from:")
    print("  https://susanqq.github.io/UTKFace/")
    print("  -> 'Aligned&Cropped Faces' ZIP File")
    print(f"  and extract images into: {os.path.abspath(image_dir)}/")
    print()

    try:
        gdown.download(gdrive_url, temp_zip, quiet=False, fuzzy=True)
    except Exception as e:
        # Try alternate known file IDs / URLs
        alt_urls = [
            "https://drive.google.com/uc?id=0BxYys69jI14kU0I1YUQyY1ZDRUE",
            "https://drive.google.com/uc?id=0BxYys69jI14kSVdWWllDMWhnN2c",
        ]
        downloaded = False
        for url in alt_urls:
            try:
                gdown.download(url, temp_zip, quiet=False, fuzzy=True)
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(f"\nAutomatic download failed: {e}")
            print(f"\nPlease download manually:")
            print(f"  1. Go to https://susanqq.github.io/UTKFace/")
            print(f"  2. Download 'Aligned&Cropped Faces' ZIP File")
            print(f"  3. Extract all .jpg images into: {os.path.abspath(image_dir)}/")
            return

    # Extract and copy only needed images
    print("Extracting...")
    os.makedirs(temp_extract, exist_ok=True)

    if temp_zip.endswith(".zip"):
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(temp_extract)
    else:
        import tarfile
        with tarfile.open(temp_zip, "r:*") as tf:
            tf.extractall(temp_extract)

    # Find and copy required images
    os.makedirs(image_dir, exist_ok=True)
    found = 0
    for root, dirs, files in os.walk(temp_extract):
        for fname in files:
            if fname in required_files:
                src = os.path.join(root, fname)
                dst = os.path.join(image_dir, fname)
                shutil.copy2(src, dst)
                found += 1

    print(f"Copied {found}/{len(required_files)} required images to {image_dir}/")

    # Cleanup temp files
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
    if os.path.exists(temp_extract):
        shutil.rmtree(temp_extract)

    if found < len(required_files):
        missing = required_files - set(os.listdir(image_dir))
        print(f"\nMissing {len(missing)} images. These may be in a different UTKFace part.")
        print("Missing files:", list(missing)[:10], "...")


if __name__ == "__main__":
    download_and_prepare()
