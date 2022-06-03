import torch
import torchvision


def save_checkpoint(state, filename="my_checkpoints.pth.tar"):
    print("========== Saving Checkpoint ==========")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("========== Loading Checkpoint ==========")
    model.load_state_dict(checkpoint["state_dict"])

# for binary prediction
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    # dice score for binary classification
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum() / ((preds + y).sum()) + 1e-8)

    print(f"{num_correct} / {num_pixels} with accuracy {(num_correct/num_pixels) * 100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")

    model.train()