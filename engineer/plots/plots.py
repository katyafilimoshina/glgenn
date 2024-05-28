import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses, val_losses, test_losses):
    """
    Plots train_losses, val_losses, test_losses
    """
    val_check_interval = len(train_losses) / len(val_losses)
    test_check_interval = len(train_losses) / len(test_losses)
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(0, len(train_losses), val_check_interval), val_losses, label='Val Loss', color='orange')
    plt.xlabel(f'Step (val was on each {int(val_check_interval)} step of training)')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(0, len(train_losses), test_check_interval), test_losses, label='Test Loss', color='green')
    plt.xlabel(f'Step (test was on each {int(test_check_interval)} step of training)')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_traintest_losses(train_losses, val_losses):
    """
    Plots train_losses and val_losses
    """
    val_check_interval = len(train_losses) / len(val_losses)
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, len(train_losses), val_check_interval), val_losses, label='Test Loss', color='green')
    plt.xlabel(f'Step (test was on each {int(val_check_interval)} step of training)')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_test_losses(train_losses, test_losses, num_to_show):
    """
    Plots the last num_to_show items of test_losses
    """
    test_check_interval = len(train_losses) / len(test_losses)
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(train_losses) - len(test_losses[-num_to_show:]) * test_check_interval, len(train_losses), test_check_interval), test_losses[-num_to_show:], label='Test Loss', color='green')
    plt.xlabel('Step of training')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()