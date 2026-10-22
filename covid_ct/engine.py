"""Training and evaluation loops."""
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return the average loss and accuracy."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)

    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    """Return the average loss and the accuracy of the model on a data loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += labels.size(0)

    return total_loss / n, correct / n


def get_predictions(model, loader, device):
    """Return the predicted labels and the true labels for a whole data loader."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting", leave=False):
            outputs = model(images.to(device))
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_labels)
