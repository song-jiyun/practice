import numpy as np
import datetime
import torch
from tqdm import tqdm

import checkpoint as cp
from plot_utils import save_training_curve

def rand_bbox(size, lam):
    # size is (N, C, H, W)
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # bbx = height (y), bby = width (x) for slicing x[:, :, bbx1:bbx2, bby1:bby2]
    bbx1 = np.clip(cy - cut_h // 2, 0, H)
    bby1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cy + cut_h // 2, 0, H)
    bby2 = np.clip(cx + cut_w // 2, 0, W)
    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    y_a = y
    y_b = y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def train(model, train_loader, optimizer, criterion, device='cpu', cutmix_prob=0.0, cutmix_alpha=1.0):
    model.train()
    total_samples = len(train_loader.dataset)
    processed = 0
    running_loss = 0.0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc='Training',
        leave=True,
        dynamic_ncols=True,
    )

    for image, label in progress:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            image, targets_a, targets_b, lam = cutmix_data(image, label, cutmix_alpha)
            output = model(image)
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            output = model(image)
            loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)
        processed += image.size(0)
        progress.set_postfix(
            loss=f'{loss.item():.4f}',
            samples=f'{processed}/{total_samples}',
            refresh=True,
        )

    progress.close()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def test(model, test_loader, criterion, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item() * image.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{ts}] Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

def training(end_epoch, model, train_loader, test_loader, optimizer, criterion, scheduler=None,
            name='training', device='cpu', cutmix_prob=0.0, cutmix_alpha=1.0):
    best_loss = float('inf')
    best_accuracy = 0.0

    today = datetime.datetime.now().strftime("%y%m%d")
    name = f'{today}_{name}_{train_loader.batch_size}_{end_epoch}'

    start_epoch, train_losses, test_losses, accuracies = cp.load_latest(model, optimizer, name, device=device)

    for epoch in range(start_epoch, end_epoch + 1):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Epoch {epoch}/{end_epoch}")

        train_loss = train(model, train_loader, optimizer, criterion, device=device, cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)
        test_loss, accuracy = test(model, test_loader, criterion, device=device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        scheduler.step() if scheduler is not None else None

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_loss = test_loss
            cp.save_best(model, name)
            cp.save_latest(model, optimizer, epoch, train_losses, test_losses, accuracies, name)
        elif accuracy == best_accuracy and test_loss < best_loss - 1e-4:
            best_loss = test_loss
            cp.save_best(model, name)
            cp.save_latest(model, optimizer, epoch, train_losses, test_losses, accuracies, name)
        elif epoch % 5 == 0:
            cp.save_latest(model, optimizer, epoch, train_losses, test_losses, accuracies, name)

        save_training_curve(train_losses, test_losses, accuracies, name)