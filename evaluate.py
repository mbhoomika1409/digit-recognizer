import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from model import DigitCNN

MEAN, STD = 0.1307, 0.3081

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------- Load test data (clean, no augmentation) ----------
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
test_loader = DataLoader(test_ds, batch_size=256)

# ---------- Load trained model ----------
model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()

# ---------- Predict on the whole test set ----------
all_preds, all_labels, all_images = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels)
        all_images.append(images)

preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
images = torch.cat(all_images)

accuracy = (preds == labels).mean()
print(f"Test accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(labels, preds, digits=4))

# ---------- Confusion matrix ----------
cm = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix (Test accuracy {accuracy * 100:.2f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------- Misclassified examples ----------
wrong = np.where(preds != labels)[0][:12]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
for ax, idx in zip(axes.flat, wrong):
    ax.imshow(images[idx].squeeze() * STD + MEAN, cmap="gray")   # undo normalization
    ax.set_title(f"True {labels[idx]} / Pred {preds[idx]}", fontsize=9)
    ax.axis("off")
for ax in axes.flat[len(wrong):]:
    ax.axis("off")
plt.suptitle("Misclassified test images")
plt.tight_layout()
plt.savefig("misclassified.png", dpi=150)

print("Saved: confusion_matrix.png, misclassified.png")