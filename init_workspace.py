"""
init_workspace.py
-----------------
Creates the directory tree expected by the other scripts.
Run once after cloning the repository.

    python init_workspace.py
"""

from pathlib import Path


def init_workspace():
    """Create every output and checkpoint directory the project needs."""
    directories = [
        "outputs/axis",
        "outputs/modified",
        "outputs/sampled",
        "outputs/iterations",
        "outputs/plots",
        "checkpoints",
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_workspace()
    print("Workspace initialised — all directories created.")
