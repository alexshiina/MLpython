"""
Contains other useful utility functions:
save_model
plot_loss_curves
"""
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def save_model(model:torch.nn.Module,
               targed_dir: str,
               model_name: str):
  """
  Saves a PyTorch model to a target directory.

  Args:
  model: Target model.
  targed_dir: Directory for saving the model to.
  model_name: A filename for the saved model. Should include
  either ".pth" or ".pt" as the file extension.
  """
  #creating directory
  targed_dir_path = Path(targed_dir)
  targed_dir.mkdir(parents = True,
                   exist_ok = True)

  #creating model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with '.pt' or '.pth'"
  model_save_path = targed_dir_path/model_name

  #saving
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.load_state_dict(),
             f=model_save_path)

#function from mrdbourke's helper_functions.py
#https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
