from models.feedback_optimizer import DirectFeedbackAlignmentOptimizer
from models.perturbation_optimizer import PerturbationOptimizer
from .evaluate_model import evaluate_model
from utils import save_training_results
import time
import torch
import psutil


def train_single_model(trainable, loss_fn, epochs, save_results=False, client_id=None, communication_round=None):
    """
    Trains a single model for a specified number of epochs, calculates relevant metrics
    during training, and optionally saves the model and/or results.

    :param trainable: An object encapsulating the model, its configurations, loaders,
        and optimizer. Must have attributes like `model`, `config`, `train_loader`,
        `val_loader`, `device`, and `optimizer`.
    :type trainable: Trainable.
    :param loss_fn: The loss function is used to calculate the loss during training.
        Defines the criterion for optimization.
    :param epochs: The number of training iterations (epochs) to perform.
    :type epochs: int
    :param save_results: Boolean flag indicating whether the results (e.g., metrics)
        should be saved after each validation iteration. Defaults to False.
    :type save_results: bool
    :param client_id: The id of the client to use for saving results. Defaults to None.
    :type client_id: int, optional
    :param communication_round: The communication round number for saving results.
    :type communication_round: int, optional
    :return: None
    """
    training_time = 0
    for t in range(epochs):
        if trainable.config.VERBOSE: print(f"--------- Epoch {t + 1} / {epochs} ----------")
        train_loss, correct, cpu = 0, 0, 0
        trainable.model.train()
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(trainable.train_loader):
            data, target = data.to(trainable.device), target.to(trainable.device)

            pred = trainable.model.forward(data)
            if isinstance(trainable.optimizer, PerturbationOptimizer):
                def closure():
                    y = trainable.model.forward(data)
                    l = loss_fn(y, target)
                    return l
                loss = trainable.optimizer.step(closure)
            elif isinstance(trainable.optimizer, DirectFeedbackAlignmentOptimizer):
                loss = loss_fn(pred, target)
                trainable.optimizer.step(loss, pred, trainable.model.inputs, trainable.model.activations)
            else:
                loss = loss_fn(pred, target)
                loss.backward()
                trainable.optimizer.step()
                trainable.optimizer.zero_grad()

            # Accumulate loss, accuracy
            train_loss += loss.item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            cpu += psutil.cpu_percent()
            if (batch_idx + 1) % 100 == 0 and trainable.config.VERBOSE:
                batch_acc = 100. * (pred.argmax(1) == target).type(torch.float).sum().item() / target.size(0)
                print(f'Batch {batch_idx + 1} | Loss: {loss.item():.4f} | Accuracy: {batch_acc:.2f}%'
                      f"{f' | ε: {trainable.privacy_engine.get_epsilon(trainable.config.DELTA)}' if trainable.config.MODEL_TYPE == 'DP' else ''}")

        epoch_time = time.time() - start_time
        training_time += epoch_time
        epoch_loss = train_loss / len(trainable.train_loader)
        epoch_accuracy = 100. * correct / len(trainable.train_loader.dataset)
        epoch_cpu = cpu / len(trainable.train_loader)
        if trainable.config.VERBOSE and t % 1 == 0:
            print(f'Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_accuracy:.2f}% | Time: {epoch_time:.3f}s | CPU usage: {epoch_cpu:.2f}%'
                  f"{f' | ε: {trainable.privacy_engine.get_epsilon(trainable.config.DELTA)}' if trainable.config.MODEL_TYPE == 'DP' else ''}")
        trainable.model.eval()
        val_results = evaluate_model(trainable, loss_fn, trainable.val_loader, True, save_results)

        if save_results:
            save_training_results(trainable.config, t, epoch_loss, epoch_accuracy, val_results[0], val_results[1], epoch_time, epoch_cpu, client_id, communication_round)
    if trainable.config.VERBOSE:
        print("-" * 20)
        print('Training Completed...')
        print(f'Total Training Time: {time.strftime("%H:%M:%S", time.gmtime(training_time))}')