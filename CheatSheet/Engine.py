"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch

    Args:
    model: PyTorch model to be trained.
    dataloader: Dataloader for the model to be trained on
    loss_fn: Pytorch loss function
    optimizer: Pytorch optimizer
    device: Target device

    Returns:
    A tuple of (train_loss,train_accuracy)
    """
    model.to(device)
    model.train()
    # setup train loss and acc
    train_loss, train_acc = 0, 0

    # loop through batches
    for batch, (X, y) in enumerate(dataloader):
        # send to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    trait_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch

    Args:
    model: PyTorch model to be tested.
    dataloader: Dataloader for the model to be tested on
    loss_fn: Pytorch loss function
    device: Target device

    Returns:
    A tuple of (test_loss,test_accuracy)
    """
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # to device
            X, y = X.to(device), y.to(device)

            # forward apss
            test_pred_logits = model(X)

            # calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        len_epoch: int,
        device: torch.device
) -> Dict[str, List]:
    """ Trains and test a Pytorch model.

      Passes a target PyTorch models through train_step() and test_step()
      functions for a number of epochs, training and testing the model
      in the same epoch loop.

      Calculates, prints and stores evaluation metrics throughout.

      Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      len_epoch: An integer indicating how often to print out epochs, ex: every 100 epochs
      device: A target device

      Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} """
    model.to(device)
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        if epoch % len_epoch == 0:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


