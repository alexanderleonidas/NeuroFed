import os
import csv
import torch
import pandas as pd
from config import FederatedConfig


def check_file_exists(config, base_filename, base_path):
    """ Find an available filename with incremental suffix if a file exists """
    new_path = base_path
    suffix_count = 1
    while os.path.isfile(new_path):
        file_name, file_ext = os.path.splitext(base_filename)
        new_path = os.path.join(config.RESULTS_PATH, f"{file_name}({suffix_count}){file_ext}")
        suffix_count += 1
    return new_path

def save_training_results(config, epoch, train_loss, train_accuracy, val_loss, val_accuracy, time_taken, cpu,client_id=None, communication_round=None):
    """
    Saves the training results into a CSV file specified by the configuration. If the result file already
    exists, creates a new file with an incremental numbered suffix like (1), (2), etc.

    :param config: Configuration object containing settings related to paths and model types.
    :type config: BaseConfig
    :param epoch: The current epoch for which results are being logged.
    :type epoch: int
    :param train_loss: The loss value of the model on the training dataset.
    :type train_loss: float
    :param train_accuracy: The accuracy value of the model on the training dataset.
    :type train_accuracy: float
    :param val_loss: The loss value of the model on the validation dataset.
    :type val_loss: float
    :param val_accuracy: The accuracy value of the model on the validation dataset.
    :type val_accuracy: float
    :param time_taken: The time taken for the epoch or communication round.
    :type time_taken: float
    :param cpu: The CPU usage during the epoch or communication round.
    :type cpu: float
    :param client_id: The identifier for the client in federated training schemes. Defaults to None.
    :type client_id: Optional[int]
    :param communication_round: The current communication round in federated training schemes. Defaults to None.
    :type communication_round: Optional[int]
    :return: None
    """
    exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
    base_filename = f"{exp_type}_{config.MODEL_TYPE}_training_results.csv"
    base_path = os.path.join(config.RESULTS_PATH, base_filename)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    results_file = check_file_exists(config, base_filename, base_path)

    # Save training results to the new file
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if client_id is None:
            writer.writerow(
                ["epoch", "model_name", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "time_taken",
                 "cpu_usage"])
            writer.writerow(
                [epoch, config.MODEL_TYPE, train_loss, train_accuracy, val_loss, val_accuracy, time_taken, cpu])
        else:
            writer.writerow(
                ["communication_round", "client", "epoch", "model_name", "train_loss", "train_accuracy", "val_loss",
                 "val_accuracy", "time_taken",
                 "cpu_usage"])
            writer.writerow(
                [communication_round, client_id, epoch, config.MODEL_TYPE, train_loss, train_accuracy, val_loss,
                 val_accuracy, time_taken, cpu])

    if results_file != base_path and config.VERBOSE:
        print(f"Created new results file: {os.path.basename(results_file)}")


def save_test_results(config, test_loss, test_accuracy, precision, recall, f1, conf_matrix):
    """
    Saves the results of a test evaluation to a CSV file. If the file already exists,
    creates a new file with an incremental numbered suffix like (1), (2), etc.

    :param config: Configuration object containing experiment and model-related settings.
    :type config: BaseConfig
    :param test_loss: Loss value computed during the test evaluation.
    :type test_loss: float
    :param test_accuracy: Accuracy value computed during the test evaluation.
    :type test_accuracy: float
    :param precision: Precision value computed during the test evaluation.
    :type precision: float
    :param recall: Recall value computed during the test evaluation.
    :type recall: float
    :param f1: F1 score computed during the test evaluation.
    :type f1: float
    :param conf_matrix: Confusion matrix as an array-like structure.
    :type conf_matrix: array-like
    :return: None
    """
    exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
    base_filename = f"{exp_type}_{config.MODEL_TYPE}_test_results.csv"
    base_path = os.path.join(config.RESULTS_PATH, base_filename)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    results_file = check_file_exists(config, base_filename, base_path)

    # Save test results to the new file
    with open(results_file, mode='a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model_name", "test_loss", "test_accuracy", "precision", "recall", "f1_score", "confusion_matrix"])
        writer.writerow([config.MODEL_TYPE, test_loss, test_accuracy, precision, recall, f1, conf_matrix.tolist()])

    if results_file != base_path and config.VERBOSE:
        print(f"Created new test results file: {os.path.basename(results_file)}")


def load_training_results(config):
    """
    Load the training results from CSV files based on the configuration provided.
    The function determines the experiment type and constructs the appropriate file path
    to load the results. It searches for all files matching the base filename pattern,
    including those with incremental suffixes like (1), (2), etc.

    :param config: Configuration object containing attributes `EXPERIMENT_TYPE`,
        `RESULTS_PATH`, and `MODEL_TYPE` needed to construct the file path
        for loading the training results.
    :type config: BaseConfig
    :return: A list of pandas DataFrames containing the loaded training results from
        all matching files, or an empty list if no files are found.
    :rtype: list[pandas.DataFrame]
    """
    exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
    base_filename = f"{exp_type}_{config.MODEL_TYPE}_training_results.csv"
    base_path = os.path.join(config.RESULTS_PATH, base_filename)

    results = []

    # Check for the base file
    if os.path.exists(base_path):
        results.append(pd.read_csv(base_path))

    # Check for files with incremental suffixes
    file_name, file_ext = os.path.splitext(base_filename)
    suffix_count = 1
    while True:
        suffixed_path = os.path.join(config.RESULTS_PATH, f"{file_name}({suffix_count}){file_ext}")
        if os.path.exists(suffixed_path):
            results.append(pd.read_csv(suffixed_path))
            suffix_count += 1
        else:
            break

    if not results:
        print(f"No results files found for {base_filename} or its variations.")

    return results

def save_model_safely(model, config, client_id=None):
    """
    Save a PyTorch model safely to a specified directory. This function ensures
    that the model's state dictionary is saved to a path determined based on
    the configuration settings. If a file with the same name already exists
    at the target location, the function will prompt the user for confirmation
    before overwriting the existing file.

    :param model: PyTorch model whose state dictionary is to be saved.
    :type model: torch.nn.Module
    :param config: Configuration object containing parameters for model saving
        such as directory path, file naming, and layer sizes.
    :type config: BaseConfig
    :param client_id: Optional client identifier for federated learning scenarios.
    :type client_id: Optional[int]
    :return: A boolean value indicating whether the model was successfully saved.
        Returns True if saving was successful, False otherwise.
    :rtype: bool
    """
    if client_id is not None:
        base_filename = str(config.LAYER_SIZES) + '_' + config.MODEL_TYPE + f"_client{client_id}.pth"
    elif client_id is None and isinstance(config, FederatedConfig):
        base_filename = str(config.LAYER_SIZES) + '_' + config.MODEL_TYPE + f"_global.pth"
    else:
        base_filename = str(config.LAYER_SIZES) + '_' + config.MODEL_TYPE + ".pth"
    base_path = os.path.join(config.SAVED_MODELS_PATH, base_filename)
    os.makedirs(config.SAVED_MODELS_PATH, exist_ok=True)
    model_path = check_file_exists(config, base_filename, base_path)

    torch.save(model.state_dict(), model_path)
    return True

def load_latest_model(trainable):
    """
    Loads the latest saved model for the given trainable object if it exists. The model
    is loaded based on the configuration of the trainable object, including layer sizes
    and model type. If no saved model is found, the method prints a message indicating
    that it is starting fresh.

    :param trainable: An object containing the model and other attributes that are required for loading the model.
    :type trainable: Trainable
    :return: None
    """
    suffix = str(trainable.config.LAYER_SIZES) + '_' +trainable.config.MODEL_TYPE + ".pth"
    model_path = os.path.join(trainable.config.SAVED_MODELS_PATH, str(suffix))
    if os.path.exists(model_path):
        trainable.model.load_state_dict(torch.load(model_path, map_location=trainable.device))
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found. Starting fresh.")