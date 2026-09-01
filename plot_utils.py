import matplotlib.pyplot as plt


def save_training_curve(train_loss, test_loss, accuracy, name):
    output_path = f"./training_curve/{name}.png"
    plt.figure(figsize=(10, 4))

    # The first point is the evaluation before any optimizer update.
    epochs = range(len(test_loss))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_loss, label="Test Loss", color="tab:blue")
    # Plot training loss on the same axes if provided.
    if train_loss is not None:
        plt.plot(epochs, train_loss, label="Train Loss", color="tab:green", linestyle="--")

    min_loss = min(test_loss)
    min_loss_epoch = test_loss.index(min_loss)
    plt.scatter(min_loss_epoch, min_loss, color="red", zorder=3)
    plt.annotate(
        f"Min Loss: {min_loss:.4f}\nEpoch {min_loss_epoch}",
        (min_loss_epoch, min_loss),
        textcoords="offset points",
        xytext=(8, 8),
        ha="left",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve - Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        accuracy,
        label="Test Accuracy",
        color="orange",
    )
    max_acc = max(accuracy)
    max_acc_epoch = accuracy.index(max_acc)
    plt.scatter(max_acc_epoch, max_acc, color="green", zorder=3)
    plt.annotate(
        f"Max Acc: {max_acc:.2f}%\nEpoch {max_acc_epoch}",
        (max_acc_epoch, max_acc),
        textcoords="offset points",
        xytext=(8, 8),
        ha="left"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Curve - Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
