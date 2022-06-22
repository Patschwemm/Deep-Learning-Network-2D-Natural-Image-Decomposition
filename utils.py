import torch
import torchvision
from pathlib import Path

path = Path.cwd()


def save_checkpoint(state, filename=(str(path) + "/models/my_checkpoints.pth.tar")):
    print("========== Saving Checkpoint ==========")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("========== Loading Checkpoint ==========")
    model.load_state_dict(checkpoint["state_dict"])
