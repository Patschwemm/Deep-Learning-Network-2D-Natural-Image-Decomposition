from pyexpat import model
import torch
from pathlib import Path

path = Path.cwd()


def save_model_param(model, modelname, filename=(str(path) + "/models/")):
    print("========== Saving Model Parameters ==========")
    torch.save(model.state_dict(), str(filename + modelname + ".pth"))


def load_model_param(model, model_file_name, filename=(str(path) + "/models/")):
    print("========== Loading Model Parameters ==========")
    model.load_state_dict(torch.load(str(filename + model_file_name + ".pth")))
