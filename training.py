import numpy as np
import torch

import checkpoint as cp
from plot_utils import save_training_curve
from schedulers import step_scheduler


def rand_bbox(size, lam):
    """Return CutMix bounding-box coordinates for an ``(N, C, H, W)`` tensor."""
    height = size[2]
    width = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_width = int(width * cut_rat)
    cut_height = int(height * cut_rat)

    center_x = np.random.randint(width)
    center_y = np.random.randint(height)

    # Slicing order is x[:, :, top:bottom, left:right].
    top = np.clip(center_y - cut_height // 2, 0, height)
    left = np.clip(center_x - cut_width // 2, 0, width)
    bottom = np.clip(center_y + cut_height // 2, 0, height)
    right = np.clip(center_x + cut_width // 2, 0, width)
    return top, left, bottom, right


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    y_a = y
    y_b = y[index]
    top, left, bottom, right = rand_bbox(x.size(), lam)
    x[:, :, top:bottom, left:right] = x[index, :, top:bottom, left:right]
    lam = 1 - ((bottom - top) * (right - left) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device="cpu",
    cutmix_prob=0.0,
    cutmix_alpha=1.0,
    callback=None,
):
    model.train()
    total_samples = len(train_loader.dataset)
    processed = 0
    running_loss = 0.0

    for batch_index, (image, label) in enumerate(train_loader):
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
        if callback is not None:
            callback(
                "train_batch_end",
                batch=batch_index + 1,
                total_batches=len(train_loader),
                loss=loss.item(),
                processed=processed,
                total_samples=total_samples,
            )

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def test(model, test_loader, criterion, device="cpu", callback=None):
    model.eval()
    test_loss = 0
    correct = 0
    processed = 0

    with torch.no_grad():
        for batch_index, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item() * image.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            processed += image.size(0)

            if callback is not None:
                callback(
                    "test_batch_end",
                    batch=batch_index + 1,
                    total_batches=len(test_loader),
                    accuracy=100.0 * correct / processed,
                    processed=processed,
                    total_samples=len(test_loader.dataset),
                )

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


def training(config, model, train_loader, test_loader, optimizer, scheduler, criterion, callback=None):
    end_epoch = config["training"]["epoch"]
    device = config["model"]["device"]

    cutmix_prob = config["cutmix"]["prob"]
    cutmix_alpha = config["cutmix"]["alpha"]

    seed = config["training"]["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    name = (
        f"{config['dataset']['dataset']}+{train_loader.batch_size}+"
        f"{config['model']['model']}+{config['optimizer']['optimizer']}+"
        f"{config['scheduler']['scheduler']}+{config['criterion']['criterion']}"
    )

    (
        start_epoch,
        best_loss,
        best_accuracy,
        train_losses,
        test_losses,
        accuracies,
    ) = cp.load_latest(model, optimizer, scheduler, name, device=device)

    if callback is not None:
        callback(
            "training_start",
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            train_batches=len(train_loader),
            test_batches=len(test_loader),
            name=name,
        )

    # Evaluate a fresh model once for an epoch-0 baseline. A checkpoint with
    # history must not receive a second baseline after resuming.
    if start_epoch == 1 and not test_losses and not accuracies:
        if callback is not None:
            callback(
                "epoch_start",
                epoch=0,
                end_epoch=end_epoch,
                train_batches=0,
                test_batches=len(test_loader),
            )

        initial_test_loss, initial_accuracy = test(
            model, test_loader, criterion, device=device, callback=callback
        )
        train_losses.append(None)
        test_losses.append(initial_test_loss)
        accuracies.append(initial_accuracy)
        best_loss = initial_test_loss
        best_accuracy = initial_accuracy
        cp.save_best(model, name)
        cp.save_latest(
            0,
            model,
            optimizer,
            scheduler,
            best_loss,
            best_accuracy,
            train_losses,
            test_losses,
            accuracies,
            config,
            name,
        )
        save_training_curve(train_losses, test_losses, accuracies, name)

        if callback is not None:
            callback(
                "epoch_end",
                epoch=0,
                train_loss=None,
                test_loss=initial_test_loss,
                accuracy=initial_accuracy,
                best_loss=best_loss,
                best_accuracy=best_accuracy,
                is_initial=True,
            )

    for epoch in range(start_epoch, end_epoch + 1):
        if callback is not None:
            callback(
                "epoch_start",
                epoch=epoch,
                end_epoch=end_epoch,
                train_batches=len(train_loader),
                test_batches=len(test_loader),
            )

        train_loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            cutmix_prob=cutmix_prob,
            cutmix_alpha=cutmix_alpha,
            callback=callback,
        )
        test_loss, accuracy = test(
            model, test_loader, criterion, device=device, callback=callback
        )

        step_scheduler(config["scheduler"], scheduler, test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_loss = test_loss
            cp.save_best(model, name)
            cp.save_latest(
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                best_accuracy,
                train_losses,
                test_losses,
                accuracies,
                config,
                name,
            )
        elif accuracy == best_accuracy and test_loss < best_loss - 1e-4:
            best_loss = test_loss
            cp.save_best(model, name)
            cp.save_latest(
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                best_accuracy,
                train_losses,
                test_losses,
                accuracies,
                config,
                name,
            )
        elif epoch % 10 == 0:
            cp.save_latest(
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                best_accuracy,
                train_losses,
                test_losses,
                accuracies,
                config,
                name,
            )

        save_training_curve(train_losses, test_losses, accuracies, name)

        if callback is not None:
            callback(
                "epoch_end",
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                accuracy=accuracy,
                best_loss=best_loss,
                best_accuracy=best_accuracy,
            )

    if callback is not None:
        callback("training_end")
