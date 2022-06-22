import torch
import torch.nn as nn
import geometry_helper
import main_network
import tqdm

def train(
    loader: torch.utils.data.Dataloader, 
    model: nn.Module, 
    optimizer: torch.optim, 
    loss_fn: function, 
    scaler: torch.autograd, 
    Device: torch.device):

    loop = tqdm(loader)
    model = model.to(device=Device)


    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=Device)
        targets = targets.float().to(device=Device)


        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())