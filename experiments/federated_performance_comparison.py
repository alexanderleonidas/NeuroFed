import time
from copy import copy
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch.nn as nn
from config import *
from data.federated_data import FederatedDataManager
from experiments.experiment_logger import ExperimentLogger
from experiments.privacy_experiment import PrivacyExperiment
from federated.server import FederatedServer


class FederatedPerformanceComparison:
    def __init__(self, configs: list[BaseConfig], save_model=False, save_results=False, plot_results=False, perform_attack=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        self.perform_attack = perform_attack
        self.configs = configs

    def run(self, num_experiments=1):
        for _ in range(num_experiments):
            loss_fn = nn.CrossEntropyLoss()
            run_id = str(int(time.time()))
            for c in self.configs:
                fed_config = FederatedConfig(c)
                logger = ExperimentLogger(fed_config, run_id=run_id)
                data = FederatedDataManager(fed_config)
                # client_sizes = [10000 for _ in range(fed_config.NUM_CLIENTS)] # If you want to specify client sizes directly uncomment this
                client_sizes = None
                if self.perform_attack:
                    full_train_loader, client_loaders, target_test_loader, shadow_loaders = data.get_privacy_experiment_shadow_loaders(client_sizes=client_sizes)
                    if fed_config.VERBOSE: print(f'Creating {fed_config.NAME} model for {c.DATASET} dataset...')
                    server = FederatedServer(fed_config, client_loaders, target_test_loader)
                    server.train_environment(logger, loss_fn, save_results=self.save_results, save_model=self.save_model)
                    server.evaluate_global_model(logger, loss_fn, save_results=self.save_results)
                    # Run the privacy experiment
                    shadow_config = copy(c)
                    attack = PrivacyExperiment(shadow_config, self.save_model, self.save_results, self.plot_results)
                    if attack.attack_config.ATTACK_TYPE == 'basic':
                        server.global_model.train_loader = full_train_loader
                        attack.run_basic_black_box_attack(server.global_model, shadow_loaders, run_id=run_id)
                    else:
                        raise NotImplementedError('Other attack types not implemented yet')
                else:
                    client_loaders, test_loader = data.get_client_loaders(client_sizes=client_sizes)
                    if fed_config.VERBOSE: print(f'Creating {fed_config.NAME} model for {c.DATASET} dataset...')
                    server = FederatedServer(fed_config, client_loaders, test_loader)
                    server.train_environment(logger, loss_fn, save_results=self.save_results, save_model=self.save_model)
                    server.evaluate_global_model(logger, loss_fn, save_results=self.save_results)

        if self.plot_results:
            self.plot_from_saved_results()

    def plot_from_saved_results(self):
        """
        Loads saved federated experiment results and generates plots comparing different model types
        across communication rounds for loss, accuracy, CPU usage, and time taken.
        Aggregates results across clients.
        """
        results = {}
        dp_addon = {}
        model_types_found = []

        print("Loading federated results...")

        for c in self.configs:
            try:
                model_type = c.MODEL_TYPE
                # Create a unique key for each model configuration, including epsilon for DP models
                epsilon_str = str(getattr(c, 'EPSILON', 'N/A')) if hasattr(c, 'EPSILON') else ''
                model_key = f"{model_type}_ε{epsilon_str}" if epsilon_str else model_type

                run_data = ExperimentLogger.load_all_runs(c)

                if run_data is None or 'training' not in run_data or run_data['training'] is None:
                    print(f"Warning: No 'training' data found for config {model_type}. Skipping.")
                    continue

                df_training = run_data['training']
                if df_training.empty:
                    print(f"Warning: Empty 'training' DataFrame for config {model_type}. Skipping.")
                    continue

                results[model_key] = df_training
                dp_addon[model_key] = f", ε={epsilon_str}" if epsilon_str else ''
                model_types_found.append(model_key)
                print(f"Successfully loaded data for {model_type}{dp_addon[model_key]}.")

            except Exception as e:
                print(f"Error loading data for config {getattr(c, 'MODEL_TYPE', 'Unknown')}: {e}")
                continue

        if not results:
            print("No valid federated results loaded. Cannot generate plots.")
            return

        # --- Plot Setup ---
        plt.style.use('seaborn-v0_8-ticks')
        color_palette = plt.get_cmap('tab10')
        colors = {model_type: color_palette(i) for i, model_type in enumerate(model_types_found)}
        default_color = 'grey'

        # Create figures using constrained_layout=True
        fig_loss, ax_loss = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_acc, ax_acc = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_cpu, ax_cpu = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig_time, ax_time = plt.subplots(figsize=(12, 7), constrained_layout=True)

        all_figures = [fig_loss, fig_acc, fig_cpu, fig_time]
        all_axes = [ax_loss, ax_acc, ax_cpu, ax_time]

        max_rounds = 0

        print("Generating federated plots...")
        # --- Plotting Loop ---
        for model_type, df_results in results.items():
            print(f"Processing {model_type}...")
            if df_results is None or df_results.empty:
                continue

            # Check for required columns
            required_cols = ['communication_round', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
                             'time_taken', 'cpu_usage']
            if not all(col in df_results.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_results.columns]
                print(f"Missing columns for {model_type}: {missing}. Skipping.")
                continue

            try:
                # Ensure communication_round is numeric
                df_results['communication_round'] = df_results['communication_round'].astype(int)

                # Group by communication round and aggregate across clients
                # For each round, we get the last epoch's results
                grouped = df_results.sort_values(['communication_round', 'client', 'epoch']) \
                    .groupby(['communication_round', 'client']) \
                    .last() \
                    .reset_index() \
                    .groupby('communication_round')

                stats = grouped.agg(
                    train_loss_mean=('train_loss', 'mean'), train_loss_std=('train_loss', 'std'),
                    val_loss_mean=('val_loss', 'mean'), val_loss_std=('val_loss', 'std'),
                    train_acc_mean=('train_accuracy', 'mean'), train_acc_std=('train_accuracy', 'std'),
                    val_acc_mean=('val_accuracy', 'mean'), val_acc_std=('val_accuracy', 'std'),
                    cpu_mean=('cpu_usage', 'mean'), cpu_std=('cpu_usage', 'std'),
                    time_mean=('time_taken', 'mean'), time_std=('time_taken', 'std'),
                    count=('client', 'size')
                ).fillna(0)

                rounds = stats.index.values
                if len(rounds) > 0:
                    current_max_round = rounds.max()
                    max_rounds = max(max_rounds, current_max_round)
                else:
                    continue  # Skip if no rounds found after grouping

                display_name = model_type.split('_ε')[0] if '_ε' in model_type else model_type
                label = f"{display_name}{dp_addon.get(model_type, '')}"
                color = colors.get(model_type, default_color)
                line_width = 1.8

                # Plot losses
                ax_loss.plot(rounds, stats['train_loss_mean'], color=color, linestyle='-', label=f'{label} - Train',
                             linewidth=line_width)
                ax_loss.fill_between(rounds, stats['train_loss_mean'] - stats['train_loss_std'],
                                     stats['train_loss_mean'] + stats['train_loss_std'], alpha=0.15, color=color,
                                     edgecolor='none')
                ax_loss.plot(rounds, stats['val_loss_mean'], color=color, linestyle='--', label=f'{label} - Val',
                             linewidth=line_width)
                ax_loss.fill_between(rounds, stats['val_loss_mean'] - stats['val_loss_std'],
                                     stats['val_loss_mean'] + stats['val_loss_std'], alpha=0.1, color=color,
                                     edgecolor='none')

                # Plot accuracies
                ax_acc.plot(rounds, stats['train_acc_mean'], color=color, linestyle='-', label=f'{label} - Train',
                            linewidth=line_width)
                ax_acc.fill_between(rounds, stats['train_acc_mean'] - stats['train_acc_std'],
                                    stats['train_acc_mean'] + stats['train_acc_std'], alpha=0.15, color=color,
                                    edgecolor='none')
                ax_acc.plot(rounds, stats['val_acc_mean'], color=color, linestyle='--', label=f'{label} - Val',
                            linewidth=line_width)
                ax_acc.fill_between(rounds, stats['val_acc_mean'] - stats['val_acc_std'],
                                    stats['val_acc_mean'] + stats['val_acc_std'], alpha=0.1, color=color,
                                    edgecolor='none')

                # Plot CPU usage
                ax_cpu.plot(rounds, stats['cpu_mean'], color=color, label=label, linewidth=line_width)
                ax_cpu.fill_between(rounds, stats['cpu_mean'] - stats['cpu_std'], stats['cpu_mean'] + stats['cpu_std'],
                                    alpha=0.15, color=color, edgecolor='none')

                # Plot time taken
                ax_time.plot(rounds, stats['time_mean'], color=color, label=label, linewidth=line_width)
                ax_time.fill_between(rounds, stats['time_mean'] - stats['time_std'],
                                     stats['time_mean'] + stats['time_std'], alpha=0.15, color=color, edgecolor='none')

            except Exception as e:
                print(f"Error processing data for {model_type}: {e}")
                continue

        print("Finalizing federated plots...")
        # --- Configure Plots ---
        plot_configs = [
            {'ax': ax_loss, 'title': 'Loss vs Communication Rounds', 'ylabel': 'Loss', 'y_locator': 0.25,
             'fig': fig_loss},
            {'ax': ax_acc, 'title': 'Accuracy vs Communication Rounds', 'ylabel': 'Accuracy (%)', 'y_locator': 10,
             'fig': fig_acc},
            {'ax': ax_cpu, 'title': 'CPU Usage vs Communication Rounds', 'ylabel': 'CPU Usage (%)', 'y_locator': 5,
             'fig': fig_cpu},
            {'ax': ax_time, 'title': 'Time per Communication Round', 'ylabel': 'Time (seconds)', 'y_locator': None,
             'fig': fig_time}
        ]

        active_figures = []  # Keep track of figures with data

        for config in plot_configs:
            ax = config['ax']
            fig = config['fig']
            if not ax.has_data():
                print(f"Skipping configuration for '{config['title']}' as no data was plotted.")
                plt.close(fig)
                continue

            active_figures.append(fig)
            ax.set_title(config['title'], fontsize=14, fontweight='bold')
            ax.set_xlabel('Communication Round', fontsize=12)
            ax.set_ylabel(config['ylabel'], fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

            # Set x-axis limits and ticks
            ax.set_xlim(left=0, right=max_rounds * 1.02 if max_rounds > 0 else 1)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True, min_n_ticks=5))

            # Set y-axis ticks if specified
            if config['y_locator']:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(config['y_locator']))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # Adjust y-limits
            if config['ax'] == ax_acc:
                current_ylim = ax.get_ylim()
                ax.set_ylim(bottom=max(0, current_ylim[0] * 0.95 - 1), top=min(105, current_ylim[1] * 1.05 + 1))
            else:
                current_ylim = ax.get_ylim()
                ax.set_ylim(bottom=max(0, current_ylim[0]), top=current_ylim[1] * 1.05 if current_ylim[1] > 0 else 1)

            # Legend placement
            legend_x_anchor = 1.01
            num_labels = len(ax.get_legend_handles_labels()[1])
            if num_labels > 0:
                ax.legend(loc='center left', bbox_to_anchor=(legend_x_anchor, 0.5), fontsize=10,
                          ncol=1 if num_labels < 10 else 2)
            else:
                print(f"No labels found for legend on '{config['title']}'.")

        # Save Plots
        if self.save_results and self.configs:
            try:
                base_results_path = Path(self.configs[0].RESULTS_PATH)
                plots_path = base_results_path / 'plots'
                dataset = getattr(self.configs[0], 'DATASET', 'unknown')
                plots_path.mkdir(parents=True, exist_ok=True)

                save_configs = [
                    {'fig': fig_loss, 'name': f'federated_loss_comparison_{dataset}.png'},
                    {'fig': fig_acc, 'name': f'federated_accuracy_comparison_{dataset}.png'},
                    {'fig': fig_cpu, 'name': f'federated_cpu_usage_comparison_{dataset}.png'},
                    {'fig': fig_time, 'name': f'federated_time_comparison_{dataset}.png'},
                ]

                for sconf in save_configs:
                    fig = sconf['fig']
                    filename = sconf['name']
                    if fig in active_figures:
                        save_path = plots_path / filename
                        print(f"Saving plot to: {save_path}")
                        fig.savefig(save_path, dpi=300)
                    else:
                        print(f"Skipping saving '{filename}' as figure has no data or was closed.")

            except Exception as e:
                print(f"Error saving plots: {e}")

        # Display Plots
        plt.show()
        print("Federated plotting finished.")
