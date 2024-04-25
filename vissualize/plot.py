import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

