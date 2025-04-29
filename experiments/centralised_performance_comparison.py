import os
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from config import *
from models.trainable import Trainable
from training.train_single_model import train_single_model
from training.evaluate_model import evaluate_model
from utils import save_model_safely, load_training_results
from data.centralised_data import CentralisedDataManager


class CentralisedPerformanceComparison:
    def __init__(self, configs: list[BaseConfig], save_model=False, save_results=False, plot_results=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        self.configs = configs

    def run(self, num_experiments=1):
        for _ in range(num_experiments):
            loss_fn = nn.CrossEntropyLoss()
            for c in self.configs:
                loaders = CentralisedDataManager(c.BATCH_SIZE)
                train_loader, val_loader, test_loader = loaders.get_loaders()
                trainable = Trainable(c, train_loader, val_loader)
                if c.VERBOSE: print(f'Creating {trainable.config.NAME} model...')
                train_single_model(trainable, loss_fn, c.EPOCHS, self.save_results)
                evaluate_model(trainable, loss_fn, test_loader)

                if self.save_model:
                    if trainable.config.VERBOSE and save_model_safely(trainable.model, trainable.config):
                        print(f'Model saved in {trainable.config.SAVED_MODELS_PATH}')

    def plot_from_saved_results(self):
        results = {}
        for c in self.configs:
            results[c.MODEL_TYPE] = load_training_results(c)

        if not any(results.values()):
            print("No results found to plot.")
            return

        # Setup plot style
        plt.style.use('ggplot')
        colors = {'BP': 'blue', 'DP': 'green', 'PB': 'red', 'DFA': 'purple'}

        # Create a figure for losses
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        # Create a figure for accuracies
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        # Create a figure for CPU usage
        fig_cpu, ax_cpu = plt.subplots(figsize=(10, 6))
        # Create a figure for time taken
        fig_time, ax_time = plt.subplots(figsize=(10, 6))

        for model_type, data_list in results.items():
            if not data_list:
                continue

            # Combine all runs for this model type
            combined_df = pd.concat(data_list, ignore_index=True)

            # Group by epoch and calculate statistics
            grouped = combined_df.groupby('epoch')

            epochs = grouped['epoch'].first().values

            # Calculate mean and std for each metric
            train_loss_mean = grouped['train_loss'].mean()
            train_loss_std = grouped['train_loss'].std()
            val_loss_mean = grouped['val_loss'].mean()
            val_loss_std = grouped['val_loss'].std()

            train_acc_mean = grouped['train_accuracy'].mean()
            train_acc_std = grouped['train_accuracy'].std()
            val_acc_mean = grouped['val_accuracy'].mean()
            val_acc_std = grouped['val_accuracy'].std()

            cpu_mean = grouped['cpu_usage'].mean()
            cpu_std = grouped['cpu_usage'].std()

            time_mean = grouped['time_taken'].mean()
            time_std = grouped['time_taken'].std()

            color = colors.get(model_type, 'black')

            # Plot losses
            ax_loss.plot(epochs, train_loss_mean, color=color, linestyle='-', label=f'{model_type} - Train')
            ax_loss.fill_between(
                epochs,
                train_loss_mean - train_loss_std,
                train_loss_mean + train_loss_std,
                alpha=0.2,
                color=color
            )
            ax_loss.plot(epochs, val_loss_mean, color=color, linestyle='--', label=f'{model_type} - Val', alpha=0.4)
            ax_loss.fill_between(
                epochs,
                val_loss_mean - val_loss_std,
                val_loss_mean + val_loss_std,
                alpha=0.1,
                color=color
            )

            # Plot accuracies
            ax_acc.plot(epochs, train_acc_mean, color=color, linestyle='-', label=f'{model_type} - Train')
            ax_acc.fill_between(
                epochs,
                train_acc_mean - train_acc_std,
                train_acc_mean + train_acc_std,
                alpha=0.2,
                color=color
            )
            ax_acc.plot(epochs, val_acc_mean, color=color, linestyle='--', label=f'{model_type} - Val', alpha=0.4)
            ax_acc.fill_between(
                epochs,
                val_acc_mean - val_acc_std,
                val_acc_mean + val_acc_std,
                alpha=0.1,
                color=color
            )

            # Plot CPU usage
            ax_cpu.plot(epochs, cpu_mean, color=color, label=f'{model_type}')
            ax_cpu.fill_between(
                epochs,
                cpu_mean - cpu_std,
                cpu_mean + cpu_std,
                alpha=0.2,
                color=color
            )

            # Plot time taken
            ax_time.plot(epochs, time_mean, color=color, label=f'{model_type}')
            ax_time.fill_between(
                epochs,
                time_mean - time_std,
                time_mean + time_std,
                alpha=0.2,
                color=color
            )

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
        if self.save_results:
            os.makedirs('results/plots', exist_ok=True)
            fig_loss.savefig('results/plots/centralised_loss_comparison.png', dpi=300, bbox_inches='tight')
            fig_acc.savefig('results/plots/centralised_accuracy_comparison.png', dpi=300, bbox_inches='tight')
            fig_cpu.savefig('results/plots/centralised_cpu_usage_comparison.png', dpi=300, bbox_inches='tight')
            fig_time.savefig('results/plots/centralised_time_comparison.png', dpi=300, bbox_inches='tight')

        plt.show()
