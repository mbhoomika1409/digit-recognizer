import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import DigitCNN

# ---------- Settings ----------
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
MEAN, STD = 0.1307, 0.3081

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------- Data ----------
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])

# Same images, two versions: augmented for training, clean for validation
train_full = datasets.MNIST("data", train=True, download=True, transform=train_tf)
val_full = datasets.MNIST("data", train=True, download=True, transform=eval_tf)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(60000, generator=g).tolist()
train_ds = Subset(train_full, perm[:54000])
val_ds = Subset(val_full, perm[54000:])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ---------- One pass over a dataset ----------
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ---------- Training ----------
if __name__ == "__main__":
    print("Using device:", device)

    model = DigitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "digit_cnn.pth")
            print("  -> saved best model")

    # ---------- Plots ----------
    epochs = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(epochs, history["train_loss"], label="Train")
    ax[0].plot(epochs, history["val_loss"], label="Validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="Train")
    ax[1].plot(epochs, history["val_acc"], label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print(f"Done. Best val accuracy: {best_val_acc:.4f}")