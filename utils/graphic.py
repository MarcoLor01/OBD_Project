import matplotlib.pyplot as plt

# Function for printing accuracy and loss


def plot_training_metrics(loss_history, accuracy_history=None):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 5))

    # Plot della Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot dell'accuratezza se disponibile
    if accuracy_history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy_history, label='Training Accuracy', marker='o')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
