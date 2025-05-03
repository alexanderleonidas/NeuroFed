import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
                logger = ExperimentLogger(c)
                if c.VERBOSE: print(f'Creating {trainable.config.NAME} model...')
                train_single_model(logger, trainable, loss_fn, save_results=self.save_results, save_model=self.save_model)
                evaluate_model(logger, trainable, loss_fn, test_loader)

        if self.plot_results:
            self.plot_from_saved_results()

    def plot_from_saved_results(self):
        """
        Loads saved experiment results and generates plots comparing different model types
        across epochs for loss, accuracy, CPU usage, and time taken.
        Uses constrained_layout for better automatic spacing with external legends.
        """
        results = {}
        dp_addon = {}
        model_types_found = []

        print("Loading results...")

        for c in self.configs:
            try:
                model_key = c.MODEL_TYPE
                run_data = ExperimentLogger.load_all_runs(c)

                if run_data is None or 'training' not in run_data or run_data['training'] is None:
                     print(f"Warning: No 'training' data found for config {c.MODEL_TYPE}. Skipping.")
                     continue

                df_training = run_data['training']
                if df_training.empty:
                     print(f"Warning: Empty 'training' DataFrame for config {c.MODEL_TYPE}. Skipping.")
                     continue

                results[model_key] = df_training
                epsilon_str = str(getattr(c, 'EPSILON', 'N/A')) if isinstance(c, DifferentialPrivacyConfig) else ''
                dp_addon[model_key] = f", ε={epsilon_str}" if epsilon_str else ''
                model_types_found.append(model_key)
                print(f"Successfully loaded data for {model_key}.")

            except Exception as e:
                print(f"Error loading data for config {getattr(c, 'MODEL_TYPE', 'Unknown')}: {e}")
                continue

        if not results:
            print("No valid results loaded. Cannot generate plots.")
            return

        # --- Plot Setup ---
        plt.style.use('seaborn-v0_8-ticks')
        color_palette = plt.get_cmap('tab10')
        colors = {model_type: color_palette(i) for i, model_type in enumerate(model_types_found)}
        default_color = 'grey'

        # Create figures using constrained_layout=True
        # NOTE: constrained_layout replaces the need for tight_layout() and subplots_adjust generally
        fig_loss, ax_loss = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_acc, ax_acc = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_cpu, ax_cpu = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_time, ax_time = plt.subplots(figsize=(12, 7), constrained_layout=True)

        all_figures = [fig_loss, fig_acc, fig_cpu, fig_time]
        all_axes = [ax_loss, ax_acc, ax_cpu, ax_time]

        max_epochs = 0

        print("Generating plots...")
        # --- Plotting Loop (identical to previous version) ---
        for model_type, df_results in results.items():
            print(f"Processing {model_type}...")
            if df_results is None or df_results.empty:
                continue
            if 'epoch' not in df_results.columns:
                if pd.api.types.is_integer_dtype(df_results.index):
                     df_results['epoch'] = df_results.index
                else:
                     print(f"Cannot determine epochs for {model_type}. Skipping this model.")
                     continue
            try:
                df_results['epoch'] = df_results['epoch'].astype(int)
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert 'epoch' column to integer for {model_type}. Check data.")
                 # Attempt to proceed if grouping still works
                 pass

            try:
                df_results_unique_epoch = df_results.drop_duplicates(subset=['epoch'], keep='last')
                grouped = df_results_unique_epoch.groupby('epoch')
                stats = grouped.agg(
                    train_loss_mean=('train_loss', 'mean'), train_loss_std=('train_loss', 'std'),
                    val_loss_mean=('val_loss', 'mean'), val_loss_std=('val_loss', 'std'),
                    train_acc_mean=('train_accuracy', 'mean'), train_acc_std=('train_accuracy', 'std'),
                    val_acc_mean=('val_accuracy', 'mean'), val_acc_std=('val_accuracy', 'std'),
                    cpu_mean=('cpu_usage', 'mean'), cpu_std=('cpu_usage', 'std'),
                    time_mean=('time_taken', 'mean'), time_std=('time_taken', 'std'),
                    count=('epoch', 'size')
                ).fillna(0)

                epochs = stats.index.values
                if len(epochs) > 0:
                   current_max_epoch = epochs.max()
                   max_epochs = max(max_epochs, current_max_epoch)
                else:
                   continue # Skip if no epochs found after grouping

                label = f"{model_type}{dp_addon.get(model_type, '')}"
                color = colors.get(model_type, default_color)
                line_width = 1.8

                # Plot losses
                if all(col in stats.columns for col in ['train_loss_mean', 'val_loss_mean']):
                    ax_loss.plot(epochs, stats['train_loss_mean'], color=color, linestyle='-', label=f'{label} - Train', linewidth=line_width)
                    ax_loss.fill_between(epochs, stats['train_loss_mean'] - stats['train_loss_std'], stats['train_loss_mean'] + stats['train_loss_std'], alpha=0.15, color=color, edgecolor='none')
                    ax_loss.plot(epochs, stats['val_loss_mean'], color=color, linestyle='--', label=f'{label} - Val', linewidth=line_width)
                    ax_loss.fill_between(epochs, stats['val_loss_mean'] - stats['val_loss_std'], stats['val_loss_mean'] + stats['val_loss_std'], alpha=0.1, color=color, edgecolor='none')

                # Plot accuracies
                if all(col in stats.columns for col in ['train_acc_mean', 'val_acc_mean']):
                    ax_acc.plot(epochs, stats['train_acc_mean'], color=color, linestyle='-', label=f'{label} - Train', linewidth=line_width)
                    ax_acc.fill_between(epochs, stats['train_acc_mean'] - stats['train_acc_std'], stats['train_acc_mean'] + stats['train_acc_std'], alpha=0.15, color=color, edgecolor='none')
                    ax_acc.plot(epochs, stats['val_acc_mean'], color=color, linestyle='--', label=f'{label} - Val', linewidth=line_width)
                    ax_acc.fill_between(epochs, stats['val_acc_mean'] - stats['val_acc_std'], stats['val_acc_mean'] + stats['val_acc_std'], alpha=0.1, color=color, edgecolor='none')

                # Plot CPU usage
                if 'cpu_mean' in stats.columns:
                    ax_cpu.plot(epochs, stats['cpu_mean'], color=color, label=label, linewidth=line_width)
                    ax_cpu.fill_between(epochs, stats['cpu_mean'] - stats['cpu_std'], stats['cpu_mean'] + stats['cpu_std'], alpha=0.15, color=color, edgecolor='none')

                # Plot time taken
                if 'time_mean' in stats.columns:
                    ax_time.plot(epochs, stats['time_mean'], color=color, label=label, linewidth=line_width)
                    ax_time.fill_between(epochs, stats['time_mean'] - stats['time_std'], stats['time_mean'] + stats['time_std'], alpha=0.15, color=color, edgecolor='none')

            except KeyError as e:
                print(f"Error processing data for {model_type}: Missing column during aggregation {e}. Check CSV headers.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred while processing/plotting data for {model_type}: {e}")
                traceback.print_exc()
                continue

        print("Finalizing plots...")
        # --- Configure Plots (after loop) ---
        plot_configs = [
            {'ax': ax_loss, 'title': 'Loss vs Epochs', 'ylabel': 'Loss', 'y_locator': 0.25, 'fig': fig_loss},
            {'ax': ax_acc, 'title': 'Accuracy vs Epochs', 'ylabel': 'Accuracy (%)', 'y_locator': 10, 'fig': fig_acc},
            {'ax': ax_cpu, 'title': 'CPU Usage vs Epochs', 'ylabel': 'CPU Usage (%)', 'y_locator': 5, 'fig': fig_cpu},
            {'ax': ax_time, 'title': 'Time per Epoch', 'ylabel': 'Time (seconds)', 'y_locator': None, 'fig': fig_time} # Auto y-ticks for time
        ]

        active_figures = [] # Keep track of figures with data

        for config in plot_configs:
            ax = config['ax']
            fig = config['fig']
            if not ax.has_data():
                print(f"Skipping configuration for '{config['title']}' as no data was plotted.")
                plt.close(fig)
                continue

            active_figures.append(fig) # Add figure to list of active ones
            ax.set_title(config['title'], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(config['ylabel'], fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

            # Set x-axis limits and ticks
            ax.set_xlim(left=0, right=max_epochs * 1.02 if max_epochs > 0 else 1) # Handle max_epochs=0 case
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True, min_n_ticks=5)) # Ensure reasonable number of ticks

            # Set y-axis ticks if specified
            if config['y_locator']:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(config['y_locator']))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # Adjust y-limits
            if config['ax'] == ax_acc:
                 current_ylim = ax.get_ylim()
                 ax.set_ylim(bottom=max(0, current_ylim[0] * 0.95 - 1), top=min(105, current_ylim[1] * 1.05 + 1)) # Slightly adjusted range
            else:
                 current_ylim = ax.get_ylim()
                 # Ensure non-negative lower bound, add some padding at top
                 ax.set_ylim(bottom=max(0, current_ylim[0]), top=current_ylim[1] * 1.05 if current_ylim[1] > 0 else 1)


            # --- Legend Placement ---
            # Adjust bbox_to_anchor x-coordinate slightly if needed (e.g., 1.01 or 1.03)
            # constrained_layout should make space, so 1.02 might be okay now, or slightly less/more.
            # Start with 1.01 for slightly closer placement.
            legend_x_anchor = 1.01
            num_labels = len(ax.get_legend_handles_labels()[1])
            if num_labels > 0:
                ax.legend(loc='center left', bbox_to_anchor=(legend_x_anchor, 0.5), fontsize=10,
                          ncol=1 if num_labels < 10 else 2) # Use 2 columns if many labels
            else:
                print(f"No labels found for legend on '{config['title']}'.")

        # --- REMOVED Layout Adjustment ---
        # constrained_layout handles this automatically. Remove the explicit tight_layout/subplots_adjust calls.
        # print("Adjusting layout and saving...") # No longer needed

        # --- Save Plots ---
        if self.save_results and self.configs:
            try:
                base_results_path = Path(self.configs[0].RESULTS_PATH)
                plots_path = base_results_path / 'plots'
                plots_path.mkdir(parents=True, exist_ok=True)

                save_configs = [
                    {'fig': fig_loss, 'name': 'centralised_loss_comparison.png'},
                    {'fig': fig_acc, 'name': 'centralised_accuracy_comparison.png'},
                    {'fig': fig_cpu, 'name': 'centralised_cpu_usage_comparison.png'},
                    {'fig': fig_time, 'name': 'centralised_time_comparison.png'},
                ]

                for sconf in save_configs:
                    fig = sconf['fig']
                    filename = sconf['name']
                    # Check if the figure is in our active list (i.e., has data and wasn't closed)
                    if fig in active_figures:
                        save_path = plots_path / filename
                        print(f"Saving plot to: {save_path}")
                        # bbox_inches='tight' can sometimes interfere slightly with constrained_layout
                        # Try saving without it first, or use bbox_inches='tight' if labels are still cut off.
                        fig.savefig(save_path, dpi=300) #, bbox_inches='tight')
                    else:
                        print(f"Skipping saving '{filename}' as figure has no data or was closed.")

            except AttributeError:
                 print("Error: Could not determine RESULTS_PATH from config. Cannot save plots.")
            except Exception as e:
                print(f"Error saving plots: {e}")
                traceback.print_exc()


        # --- Display Plots ---
        plt.show()
        print("Plotting finished.")
