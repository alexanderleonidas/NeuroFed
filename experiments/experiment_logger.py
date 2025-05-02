import os
import csv
import time
import fnmatch
import re
import torch
import pandas as pd
from config import FederatedConfig


class ExperimentLogger:
    """
    A class that manages machine learning experiments, including tracking metrics,
    saving/loading models, and organizing results. This class centralizes experiment
    management functions and maintains consistency across different experiment runs.

    Attributes:
        config (BaseConfig): Configuration object with experiment settings
        run_id (int): Unique identifier for the current experiment run
        is_federated (bool): Whether this is a federated learning experiment
        verbose (bool): Whether to print detailed information during operations
        results_path (str): Path to the directory where results are stored
        models_path (str): Path to the directory where models are stored
        layer_dir (str): Path to the directory for the current layer configuration
        exp_type (str): Type of experiment (e.g., 'Centralised', 'Federated')
        model_type (str): Type of model being used in the experiment
    """

    def __init__(self, config, run_id=None):
        """
        Initialize the ExperimentManager with a configuration and optional run ID.

        :param config: Configuration object containing experiment settings
        :type config: BaseConfig
        :param run_id: Unique identifier for the experiment run, generated if None
        :type run_id: Optional[int]
        """
        self.config = config
        self.run_id = run_id if run_id is not None else self._generate_run_id()
        self.is_federated = isinstance(config, FederatedConfig)
        self.verbose = getattr(config, 'VERBOSE', False)

        # Set up paths
        self.results_path = config.RESULTS_PATH
        self.models_path = config.SAVED_MODELS_PATH
        self.layer_dir = os.path.join(self.models_path, f"layers_{str(config.LAYER_SIZES).replace(', ', '_')}")

        # Ensure directories exist
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.layer_dir, exist_ok=True)

        # Set experiment and model types
        self.exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
        self.model_type = config.MODEL_TYPE

        if self.verbose:
            print(f"Initialized experiment logger for run {self.run_id}")

    def _generate_run_id(self):
        """
        Generate a unique run ID based on the current timestamp.

        :return: Integer timestamp to serve as the run ID
        :rtype: int
        """
        return int(time.time())

    def _get_results_filepath(self, file_type="training"):
        """
        Construct the file path for storing results of the specified type.

        :param file_type: Type of results ('training' or 'test')
        :type file_type: str
        :return: Full path to the results file
        :rtype: str
        """
        filename = f"{self.exp_type}_{self.model_type}_run{self.run_id}_{file_type}_results.csv"
        return os.path.join(self.results_path, filename)

    def _get_model_filepath(self, client_id=None):
        """
        Construct the file path for storing a model.

        :param client_id: Client ID for federated learning (if applicable)
        :type client_id: Optional[int]
        :return: Full path to the model file
        :rtype: str
        """
        if client_id is not None:
            filename = f"{self.model_type}_run{self.run_id}_client{client_id}.pth"
        elif self.is_federated:
            filename = f"{self.model_type}_run{self.run_id}_global.pth"
        else:
            filename = f"{self.model_type}_run{self.run_id}.pth"

        return os.path.join(self.layer_dir, filename)

    def save_training_results(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy,
                              time_taken, cpu, client_id=None, communication_round=None):
        """
        Save training results to a CSV file for the current experiment run.

        :param epoch: Current training epoch
        :type epoch: int
        :param train_loss: Loss value on the training dataset
        :type train_loss: float
        :param train_accuracy: Accuracy on the training dataset
        :type train_accuracy: float
        :param val_loss: Loss value on the validation dataset
        :type val_loss: float
        :param val_accuracy: Accuracy on the validation dataset
        :type val_accuracy: float
        :param time_taken: Time taken for the epoch or communication round
        :type time_taken: float
        :param cpu: CPU usage during the epoch or communication round
        :type cpu: float
        :param client_id: Identifier for the client in federated training
        :type client_id: Optional[int]
        :param communication_round: Current communication round in federated training
        :type communication_round: Optional[int]
        :return: Self for method chaining
        :rtype: ExperimentManager
        """
        results_file = self._get_results_filepath("training")
        is_first_entry = not os.path.exists(results_file)

        with open(results_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if is_first_entry:
                if client_id is None:
                    writer.writerow(
                        ["run_id", "epoch", "model_name", "train_loss", "train_accuracy",
                         "val_loss", "val_accuracy", "time_taken", "cpu_usage"])
                else:
                    writer.writerow(
                        ["run_id", "communication_round", "client", "epoch", "model_name",
                         "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                         "time_taken", "cpu_usage"])

            # Write the data row
            if client_id is None:
                writer.writerow(
                    [self.run_id, epoch, self.model_type, train_loss, train_accuracy,
                     val_loss, val_accuracy, time_taken, cpu])
            else:
                writer.writerow(
                    [self.run_id, communication_round, client_id, epoch, self.model_type,
                     train_loss, train_accuracy, val_loss, val_accuracy, time_taken, cpu])

        if is_first_entry and self.verbose:
            print(f"Created new training results file for run {self.run_id}: {os.path.basename(results_file)}")

        return self

    def save_test_results(self, test_loss, test_accuracy, precision, recall, f1, conf_matrix):
        """
        Save test evaluation results to a CSV file for the current experiment run.

        :param test_loss: Loss value on the test dataset
        :type test_loss: float
        :param test_accuracy: Accuracy on the test dataset
        :type test_accuracy: float
        :param precision: Precision metric from the evaluation
        :type precision: float
        :param recall: Recall metric from the evaluation
        :type recall: float
        :param f1: F1 score from the evaluation
        :type f1: float
        :param conf_matrix: Confusion matrix as an array-like structure
        :type conf_matrix: array-like
        :return: Self for method chaining
        :rtype: ExperimentManager
        """
        results_file = self._get_results_filepath("test")
        is_first_entry = not os.path.exists(results_file)

        with open(results_file, mode='a', newline="") as f:
            writer = csv.writer(f)
            if is_first_entry:
                writer.writerow(
                    ["run_id", "model_name", "test_loss", "test_accuracy", "precision",
                     "recall", "f1_score", "confusion_matrix"])

            writer.writerow([self.run_id, self.model_type, test_loss, test_accuracy,
                             precision, recall, f1, conf_matrix.tolist()])

        if is_first_entry and self.verbose:
            print(f"Created new test results file for run {self.run_id}: {os.path.basename(results_file)}")

        return self

    def save_model(self, model, client_id=None):
        """
        Save a PyTorch model for the current experiment run.

        :param model: PyTorch model to save
        :type model: torch.nn.Module
        :param client_id: Identifier for the client in federated training
        :type client_id: Optional[int]
        :return: Self for method chaining
        :rtype: ExperimentManager
        """
        model_path = self._get_model_filepath(client_id)
        torch.save(model.state_dict(), model_path)

        if self.verbose:
            print(f"Saved model for run {self.run_id} to {model_path}")

        return self

    def load_model(self, model, client_id=None):
        """
        Load a saved model for the current experiment run into the provided model.

        :param model: PyTorch model to load the state into
        :type model: torch.nn.Module
        :param client_id: Identifier for the client in federated training
        :type client_id: Optional[int]
        :return: True if model was loaded successfully, False otherwise
        :rtype: bool
        """
        model_path = self._get_model_filepath(client_id)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

            if self.verbose:
                print(f"Loaded model from {model_path}")

            return True
        else:
            if self.verbose:
                print(f"No saved model found at {model_path}")

            return False

    def load_training_results(self):
        """
        Load training results for the current experiment run.

        :return: DataFrame containing the training results
        :rtype: pandas.DataFrame
        """
        results_file = self._get_results_filepath("training")

        if os.path.exists(results_file):
            return pd.read_csv(results_file)
        else:
            if self.verbose:
                print(f"No training results file found for run {self.run_id}")

            return pd.DataFrame()

    def load_test_results(self):
        """
        Load test results for the current experiment run.

        :return: DataFrame containing the test results
        :rtype: pandas.DataFrame
        """
        results_file = self._get_results_filepath("test")

        if os.path.exists(results_file):
            return pd.read_csv(results_file)
        else:
            if self.verbose:
                print(f"No test results file found for run {self.run_id}")

            return pd.DataFrame()

    @classmethod
    def find_latest_run(cls, config, client_id=None):
        """
        Find the latest experiment run ID for the given configuration.

        :param config: Configuration object with experiment settings
        :type config: BaseConfig
        :param client_id: Identifier for the client in federated training
        :type client_id: Optional[int]
        :return: Latest run ID if found, None otherwise
        :rtype: Optional[int]
        """
        is_federated = isinstance(config, FederatedConfig)
        exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
        model_type = config.MODEL_TYPE
        layer_dir = os.path.join(config.SAVED_MODELS_PATH,
                                 f"layers_{str(config.LAYER_SIZES).replace(', ', '_')}")

        if not os.path.exists(layer_dir):
            return None

        # Set up a pattern based on client_id
        if client_id is not None:
            pattern = f"{model_type}_run*_client{client_id}.pth"
        elif is_federated:
            pattern = f"{model_type}_run*_global.pth"
        else:
            pattern = f"{model_type}_run*.pth"

        # Find all matching files
        all_files = os.listdir(layer_dir)
        matching_files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]

        if not matching_files:
            return None

        # Extract run IDs and find the latest
        run_ids = []
        for filename in matching_files:
            match = re.search(r'run(\d+)', filename)
            if match:
                run_ids.append(int(match.group(1)))

        return max(run_ids) if run_ids else None

    @classmethod
    def from_latest_run(cls, config, client_id=None):
        """
        Create an ExperimentManager instance from the latest run for the given configuration, if one is notfound, a new
        instance is created.

        :param config: Configuration object with experiment settings
        :type config: BaseConfig
        :param client_id: Identifier for the client in federated training
        :type client_id: Optional[int]
        :return: ExperimentManager instance or None if no previous runs found
        :rtype: Optional[ExperimentManager]
        """
        run_id = cls.find_latest_run(config, client_id)

        if run_id is not None:
            return cls(config, run_id)
        else:
            if getattr(config, 'VERBOSE', False):
                print("No previous runs found. A new ExperimentManager will be created.")

            return cls(config)

    @classmethod
    def load_all_runs(cls, config):
        """
        Load training and test results for all runs of a given configuration.

        :param config: Configuration object with experiment settings
        :type config: BaseConfig
        :return: Dictionary with 'training' and 'test' DataFrames containing all runs
        :rtype: dict
        """
        exp_type = getattr(config, 'EXPERIMENT_TYPE', 'Centralised')
        model_type = config.MODEL_TYPE
        results_path = config.RESULTS_PATH

        if not os.path.exists(results_path):
            return {'training': pd.DataFrame(), 'test': pd.DataFrame()}

        # Patterns for training and test results
        training_pattern = f"{exp_type}_{model_type}_run*_training_results.csv"
        test_pattern = f"{exp_type}_{model_type}_run*_test_results.csv"

        # Find and load all files
        all_files = os.listdir(results_path)

        # Process training files
        training_files = [f for f in all_files if fnmatch.fnmatch(f, training_pattern)]
        training_dfs = []

        for filename in training_files:
            file_path = os.path.join(results_path, filename)
            training_dfs.append(pd.read_csv(file_path))

        # Process test files
        test_files = [f for f in all_files if fnmatch.fnmatch(f, test_pattern)]
        test_dfs = []

        for filename in test_files:
            file_path = os.path.join(results_path, filename)
            test_dfs.append(pd.read_csv(file_path))

        return {
            'training': pd.DataFrame(pd.concat(training_dfs, ignore_index=True)) if training_dfs else pd.DataFrame(),
            'test': pd.DataFrame(pd.concat(test_dfs, ignore_index=True)) if test_dfs else pd.DataFrame()
        }