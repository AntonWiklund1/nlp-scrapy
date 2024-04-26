import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(actuals, predictions, classes, title='Confusion Matrix', file_name='./results/cm/confusion_matrix.png'):
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{title}')
    plt.savefig(f'{file_name}')

def plot_learning_curve(average_train_losses, average_val_losses, title='Learning Curve', file_name='./results/learning_curve.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(average_train_losses, label='Average Training Loss')
    plt.plot(average_val_losses, label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{file_name}')

def plot_gradients(fc1_norms, fc2_norms, title, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(fc1_norms, label='FC1 Gradient Norms')
    plt.plot(fc2_norms, label='FC2 Gradient Norms')
    plt.title(title)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.savefig(file_path)
    plt.close()

def plot_ud_ratios(ud_ratio_history, filename):
    plt.figure(figsize=(20, 4))
    for i, ud in enumerate(zip(*ud_ratio_history)):  # Transpose the list of lists
        plt.plot(ud, label=f'weight param {i}')
    plt.axhline(y=-3, color='k', linestyle='--', label='Target Ratio (-3)') # Horizontal line at log10(1e-3)

    plt.xlabel('Training Step')
    plt.ylabel('Log10(Update-to-Data Ratio)')
    plt.title('Log-Scale Update-to-Data Ratios for Model Weights Over Training')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



