import os
import matplotlib.pyplot as plt
import torch.nn as nn
from config import *
from experiments.experiment_logger import ExperimentLogger
from models.trainable import Trainable
from training.train_single_model import train_single_model
from training.evaluate_model import evaluate_model
from data.centralised_data import CentralisedDataManager


class CentralisedPerformanceComparison:
    def __init__(self, configs: list[BaseConfig], save_model=False, save_results=False, plot_results=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        if configs is None or len(configs) == 0:
            raise ValueError("No configurations provided for the experiment. Must provide a list of configuration(s).")
        self.configs = configs

    def run(self, num_experiments=1):
        for _ in range(num_experiments):
            loss_fn = nn.CrossEntropyLoss()
            for c in self.configs:
                loaders = CentralisedDataManager(c.BATCH_SIZE)
                train_loader, val_loader, test_loader = loaders.get_loaders()
                trainable = Trainable(c, train_loader, val_loader)
                logger = ExperimentLogger.from_latest_run(c)
                if c.VERBOSE: print(f'Creating {trainable.config.NAME} model...')
                train_single_model(logger, trainable, loss_fn, save_results=self.save_results, save_model=self.save_model)
                evaluate_model(logger, trainable, loss_fn, test_loader)

    def plot_from_saved_results(self):
        results = {}
        for c in self.configs:
            results[c.MODEL_TYPE] = ExperimentLogger.load_all_runs(c)['training']

        if not any(df is not None and not df.empty for df in results.values()):
            print("No results found to plot.")
            return

        # Setup plot style
        plt.style.use('ggplot')
        colors = {'BP': 'blue', 'DP': 'green', 'PB': 'red', 'DFA': 'purple'}

        # Create figures
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        fig_cpu, ax_cpu = plt.subplots(figsize=(10, 6))
        fig_time, ax_time = plt.subplots(figsize=(10, 6))

        for model_type, df_results in results.items():
            # Skip if no data or empty DataFrame for this model type
            if df_results is None or df_results.empty:
                print(f"Skipping plotting for {model_type} due to missing or empty data.")
                continue

            # Check if 'epoch' column exists
            if 'epoch' not in df_results.columns:
                print(f"Warning: 'epoch' column not found in data for {model_type}. Cannot group by epoch.")
                continue

            # Group by epoch and calculate statistics
            try:
                grouped = df_results.groupby('epoch')

                # Ensure required columns exist before accessing them
                required_cols = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'cpu_usage', 'time_taken']
                if not all(col in df_results.columns for col in required_cols):
                    print(
                        f"Warning: Missing one or more required columns {required_cols} for {model_type}. Skipping some plots.")
                    # Decide how to handle missing columns - skip model, skip plot, plot available etc.
                    # Here we'll try to plot what's available, guarding each calculation.

                epochs = grouped[
                    'epoch'].first().values  # .index could also work if 'epoch' is the index after grouping

                color = colors.get(model_type, 'black')  # Default to black if model_type not in colors dict

                # --- Plotting with checks for column existence ---

                # Plot losses
                if 'train_loss' in df_results.columns and 'val_loss' in df_results.columns:
                    train_loss_mean = grouped['train_loss'].mean()
                    train_loss_std = grouped['train_loss'].std().fillna(0)  # Handle NaN std for single data points
                    val_loss_mean = grouped['val_loss'].mean()
                    val_loss_std = grouped['val_loss'].std().fillna(0)

                    ax_loss.plot(epochs, train_loss_mean, color=color, linestyle='-', label=f'{model_type} - Train')
                    ax_loss.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                                         alpha=0.2, color=color)
                    ax_loss.plot(epochs, val_loss_mean, color=color, linestyle='--', label=f'{model_type} - Val',
                                 alpha=0.4)
                    ax_loss.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.1,
                                         color=color)
                else:
                    print(f"Skipping loss plot for {model_type} due to missing loss columns.")

                # Plot accuracies
                if 'train_accuracy' in df_results.columns and 'val_accuracy' in df_results.columns:
                    train_acc_mean = grouped['train_accuracy'].mean()
                    train_acc_std = grouped['train_accuracy'].std().fillna(0)
                    val_acc_mean = grouped['val_accuracy'].mean()
                    val_acc_std = grouped['val_accuracy'].std().fillna(0)

                    ax_acc.plot(epochs, train_acc_mean, color=color, linestyle='-', label=f'{model_type} - Train')
                    ax_acc.fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                                        alpha=0.2, color=color)
                    ax_acc.plot(epochs, val_acc_mean, color=color, linestyle='--', label=f'{model_type} - Val',
                                alpha=0.4)
                    ax_acc.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.1,
                                        color=color)
                else:
                    print(f"Skipping accuracy plot for {model_type} due to missing accuracy columns.")

                # Plot CPU usage
                if 'cpu_usage' in df_results.columns:
                    cpu_mean = grouped['cpu_usage'].mean()
                    cpu_std = grouped['cpu_usage'].std().fillna(0)

                    ax_cpu.plot(epochs, cpu_mean, color=color, label=f'{model_type}')
                    ax_cpu.fill_between(epochs, cpu_mean - cpu_std, cpu_mean + cpu_std, alpha=0.2, color=color)
                else:
                    print(f"Skipping CPU usage plot for {model_type} due to missing cpu_usage column.")

                # Plot time taken
                if 'time_taken' in df_results.columns:
                    time_mean = grouped['time_taken'].mean()
                    time_std = grouped['time_taken'].std().fillna(0)

                    ax_time.plot(epochs, time_mean, color=color, label=f'{model_type}')
                    ax_time.fill_between(epochs, time_mean - time_std, time_mean + time_std, alpha=0.2, color=color)
                else:
                    print(f"Skipping time taken plot for {model_type} due to missing time_taken column.")

            except KeyError as e:
                print(f"Error processing data for {model_type}: Missing column {e}. Skipping this model type.")
                continue
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing data for {model_type}: {e}. Skipping this model type.")
                continue

        # Configure loss plot
        ax_loss.set_title('Loss vs Epochs')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # Configure accuracy plot
        ax_acc.set_title('Accuracy vs Epochs')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

        # Configure CPU usage plot
        ax_cpu.set_title('CPU Usage vs Epochs')
        ax_cpu.set_xlabel('Epoch')
        ax_cpu.set_ylabel('CPU Usage (%)')
        ax_cpu.legend()
        ax_cpu.grid(True)

        # Configure time taken plot
        ax_time.set_title('Time per Epoch')
        ax_time.set_xlabel('Epoch')
        ax_time.set_ylabel('Time (seconds)')
        ax_time.legend()
        ax_time.grid(True)

        # Save plots if needed
        if self.save_results and self.configs:  # Check if config list is not empty
            # Ensure a base results path exists
            base_results_path = self.configs[0].RESULTS_PATH
            plots_path = os.path.join(base_results_path, 'plots')  # Use os.path.join for compatibility
            os.makedirs(plots_path, exist_ok=True)

            try:
                if fig_loss.axes[0].has_data():  # Check if a plot has data before saving
                    fig_loss.savefig(os.path.join(plots_path, 'centralised_loss_comparison.png'), dpi=300,
                                     bbox_inches='tight')
                if fig_acc.axes[0].has_data():
                    fig_acc.savefig(os.path.join(plots_path, 'centralised_accuracy_comparison.png'), dpi=300,
                                    bbox_inches='tight')
                if fig_cpu.axes[0].has_data():
                    fig_cpu.savefig(os.path.join(plots_path, 'centralised_cpu_usage_comparison.png'), dpi=300,
                                    bbox_inches='tight')
                if fig_time.axes[0].has_data():
                    fig_time.savefig(os.path.join(plots_path, 'centralised_time_comparison.png'), dpi=300,
                                     bbox_inches='tight')
            except Exception as e:
                print(f"Error saving plots: {e}")

        plt.show()
