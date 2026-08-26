from pathlib import Path
import os
import torch

def path(path):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, path)

def save_latest(epoch, model, optimizer, scheduler, best_loss, best_accuracy, train_losses, test_losses, test_accuracies, config, name):
    checkpoint = {
        'epoch': epoch,

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),

        'best_loss': best_loss,
        'best_accuracy': best_accuracy,

        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,

        'config': config
    }

    target = path(name+'_latest.pt')
    torch.save(checkpoint, target)
    print(f'Saved checkpoint to {target}')

def save_best(model, name):
    target = path(name+'_model.pt')
    torch.save(model.state_dict(), target)
    print(f'Saved best model to {target}')

def load_latest(model, optimizer, scheduler, name, device='cpu'):
    target = path(name+'_latest.pt')
    if os.path.exists(target):
        checkpoint = torch.load(target, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        test_accuracies = checkpoint.get('test_accuracies', [])
        config = checkpoint.get('config')
        epoch = checkpoint.get('epoch', 0) + 1
        print(f'Loaded checkpoint from {target}, resuming from epoch {epoch}')
        return epoch, best_loss, best_accuracy, train_losses, test_losses, test_accuracies
    else:
        return 1, float('inf'), 0.0, [], [], []

def load_best(model, name, device='cpu'):
    target = path(name+'_model.pt')
    if os.path.exists(target):
        model.load_state_dict(torch.load(target, map_location=device))
        print(f'Loaded best model from {target}')
    else:
        print(f'No best model found at {target}, starting from scratch')