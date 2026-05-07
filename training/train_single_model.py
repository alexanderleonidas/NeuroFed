import warnings
from config import FederatedConfig, AttackConfig
from models.feedback_optimizer import DirectFeedbackAlignmentOptimizer
from models.perturbation_optimizer import PerturbationOptimizer
from models.trainable import Trainable
from models.decision_tree import DecisionTreeWrapper
from .evaluate_model import evaluate_model
import time
import torch
from torch import linalg
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score

def train_single_model(logger, trainable, loss_fn, client_id=None, communication_round=None, early_stop=False, save_results=False, save_model=False):
    """
    Trains a single model, calculates relevant metrics during training, and optionally saves the model and/or results.

    :param logger: An object encapsulating the experiment logger.
    :type logger: ExperimentLogger.
    :param trainable: An object encapsulating the model, its configurations, loaders,
        and optimizer. Must have attributes like `model`, `config`, `train_loader`,
        `val_loader`, `device`, and `optimizer`.
    :type trainable: Trainable.
    :param loss_fn: The loss function is used to calculate the loss during training.
        Defines the criterion for optimization.
    :param client_id: The id of the client to use for saving results. Defaults to None.
    :type client_id: int, optional.
    :param communication_round: The communication round number for saving results.
    :type communication_round: int, optional.
    :param early_stop: Boolean flag indicating whether to apply early stopping based on validation accuracy.
    :type early_stop: bool.
    :param save_results: Boolean flag indicating whether the results (e.g., metrics)
        should be saved after each validation iteration. Defaults to False.
    :type save_results: bool.
    :param save_model: Boolean flag indicating whether the model should be saved
    :type save_model: bool.
    :return: None.
    """
    epochs = trainable.config.EPOCHS
    start_epoch = 0
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    if logger is None:
        pass
    else:
        # Check if we're continuing training else create a new run
        check_logger = logger.from_latest_run(trainable.config)
        if check_logger is not None:
            logger = check_logger
            training_df = logger.load_training_results()
            if not training_df.empty:
                start_epoch = training_df['epoch'].max() + 1
                if start_epoch >= epochs:
                    logger.create_new_run()
                    start_epoch = 0
                else:
                    print(f"Continuing from epoch {start_epoch}")

        # Try to load a saved model for the current run
        model_loaded = logger.load_model(trainable.model, client_id=client_id if client_id is not None else None)
        if model_loaded:
            print(f"Continuing training from run {logger.run_id}")
        # else:
        #     print(f"Starting new training run {logger.run_id}")

    if (not isinstance(trainable.config, FederatedConfig) or not isinstance(trainable.config, AttackConfig)) and client_id is None:
        eps = "Epsilon" if hasattr(trainable.config, 'EPSILON') else ""
        print("Epoch", "Loss", "Accuracy (%)", "Precision", "Recall", "F1 Score", "Time (s)", "CPU (%)", eps, sep="\t")
        print(100 * "-")
    training_time = 0
    for t in range(start_epoch, epochs, 1):
        train_loss, correct, cpu = 0, 0, 0
        trainable.model.train()
        start_time = time.time()
        all_labels = []
        all_predictions = []
        # grad_v = []

        for batch_idx, (data, target) in enumerate(trainable.train_loader):
            data, target = data.to(trainable.device), target.to(trainable.device)

            logits = trainable.model.forward(data)
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                logits = logits.squeeze()
                predictions = torch.round(torch.sigmoid(logits.squeeze()))
            elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                predictions = torch.softmax(logits, dim=1).argmax(dim=1)
            else:
                raise NotImplementedError

            if isinstance(trainable.optimizer, PerturbationOptimizer):
                def closure():
                    y = trainable.model.forward(data)
                    l = loss_fn(y, target)
                    return l
                loss = trainable.optimizer.step(closure)
            elif isinstance(trainable.optimizer, DirectFeedbackAlignmentOptimizer):
                loss = loss_fn(logits, target)
                # grads = [p.grad for p in trainable.model.parameters() if p.grad is not None]
                # grads = torch.cat([g.flatten() for g in grads])
                # v = linalg.vector_norm(grads, ord=2)
                # grad_v.append(v.item())
                # Clip the gradients if needed
                # torch.nn.utils.clip_grad_norm_(trainable.model.parameters(), max_norm=7)
                # torch.nn.utils.clip_grad_value_(trainable.model.parameters(), clip_value=1.0)

                trainable.optimizer.step(loss, logits, trainable.model.inputs, trainable.model.activations)
            else:
                loss = loss_fn(logits, target)
                loss.backward()
                trainable.optimizer.step()
                trainable.optimizer.zero_grad()

            # Accumulate loss, accuracy, cpu usage
            train_loss += loss.item()
            cpu += psutil.cpu_percent()
            correct += torch.eq(predictions, target).sum().item()
            all_labels.extend(target.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())

        epoch_time = time.time() - start_time
        training_time += epoch_time
        epoch_loss = train_loss / len(trainable.train_loader)
        epoch_accuracy = 100. * correct / len(trainable.train_loader.dataset)
        epoch_cpu = cpu / len(trainable.train_loader)
        epoch_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        if trainable.config.VERBOSE and t % 1 == 0:
            epsilon = f'{trainable.privacy_engine.get_epsilon(trainable.config.DELTA):.2f}' if trainable.config.MODEL_TYPE=='DP' else ""
            if isinstance(trainable.config, FederatedConfig):
                print(f'{communication_round}', f'{client_id}', t+1, f'{epoch_loss:.4f}', f'{epoch_accuracy:.2f}', f'{epoch_precision:.2f}',
                      f'{epoch_recall:.2f}', f'{epoch_f1:.2f}', f'{epoch_time:.3f}', f'{epoch_cpu:.2f}', epsilon, sep="\t\t")
            elif client_id is not None:
                print(f'{client_id}', t+1, f'{epoch_loss:.4f}', f'{epoch_accuracy:.2f}', f'{epoch_precision:.2f}',
                      f'{epoch_recall:.2f}', f'{epoch_f1:.2f}', f'{epoch_time:.3f}', f'{epoch_cpu:.2f}', epsilon, sep="\t\t")
            else:
                print(t+1, f'{epoch_loss:.4f}', f'{epoch_accuracy:.2f}', f'{epoch_precision:.2f}',
                      f'{epoch_recall:.2f}', f'{epoch_f1:.2f}', f'{epoch_time:.3f}', f'{epoch_cpu:.2f}', epsilon, sep="\t\t")

        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha", category=UserWarning)
        if trainable.val_loader is None and trainable.test_loader is None:
            val_results = (None,None,None,None,None,None)
        else:
            validation_loader = trainable.val_loader if trainable.val_loader is not None else trainable.test_loader
            val_results = evaluate_model(logger, trainable, loss_fn, validation_loader, True, save_results)

        if save_model:
            logger.save_model(trainable.model, client_id)
        if save_results:
            logger.save_training_results(t, epoch_loss, epoch_accuracy, val_results[0], val_results[1], epoch_time,
                                         epoch_cpu, epoch_precision, epoch_recall, epoch_f1, client_id, communication_round)

        # Early stopping check
        if early_stop and (trainable.val_loader is not None or trainable.test_loader is not None):
            if val_results[1] > best_val_acc:
                best_val_acc = val_results[1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and trainable.config.VERBOSE:
                    print(f"Early stopping at epoch {t+1}")
                    break

    if trainable.config.VERBOSE and not isinstance(trainable.config, FederatedConfig) and not isinstance(trainable.config, AttackConfig):
        print('Training Completed...')
        print(f'Total Training Time: {time.strftime("%H:%M:%S", time.gmtime(training_time))}')
        print("-" * 30)