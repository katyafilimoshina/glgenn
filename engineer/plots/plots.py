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


def plot_losses_compare(train_losses_list, val_losses_list, model_labels):
    """
    Plots the training and validation losses for multiple models side by side.

    :param train_losses_list: List of lists, where each sublist contains training losses for a model.
    :param val_losses_list: List of lists, where each sublist contains validation losses for a model.
    :param model_labels: List of strings, where each string is a label for a model.
    """
    def plot_loss_c(ax, steps, losses_list, title, model_labels, test=False):
        for losses, label in zip(losses_list, model_labels):
            ax.plot(steps, losses, label=label)
        if test:
          ax.set_xlabel(f'Step (test was on each {int(val_check_interval)} step of training)')
        else:
          ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    val_check_interval = len(train_losses_list[0]) / len(val_losses_list[0])
    steps = range(len(train_losses_list[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_loss_c(ax1, steps, train_losses_list, 'Train Loss', model_labels)
    plot_loss_c(ax2, np.arange(0, len(train_losses_list[0]), val_check_interval), val_losses_list, 'Test Loss', model_labels, test=True)
    plt.tight_layout()
    plt.show()