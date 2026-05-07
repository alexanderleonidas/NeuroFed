########### Import Experiments ###########
from config import *
from experiments.centralised_performance_comparison import CentralisedPerformanceComparison
from experiments.federated_performance_comparison import FederatedPerformanceComparison

def main():
    ## Choose to save models and results and plot results
    save_model = False
    save_results = False
    plot_results = False # To plot results, run the experiment with save_results = True
    perform_attack = True
    num_experiments = 1

    ## Choose which configurations to run
    # dp_config1 = DifferentialPrivacyConfig(EPSILON=0.4)
    # dp_config2 = DifferentialPrivacyConfig(EPSILON=0.7)
    # dp_config3 = DifferentialPrivacyConfig(EPSILON=1)
    # configs = [BackpropagationConfig(), DirectFeedbackAlignmentConfig(), PerturbationConfig(), dp_config3, dp_config2, dp_config1]
    # configs = [BackpropagationConfig(), DirectFeedbackAlignmentConfig(), PerturbationConfig(), DifferentialPrivacyConfig()]
    # configs = [dp_config1, dp_config2, dp_config3]
    configs = [BackpropagationConfig()]
    # configs = [DirectFeedbackAlignmentConfig()]
    # configs = [PerturbationConfig()]

    ########### Choose Experiment ###########
    experiment = CentralisedPerformanceComparison(configs, save_model, save_results, plot_results, perform_attack)

    # experiment = FederatedPerformanceComparison(configs, save_model, save_results, plot_results, perform_attack)

    ########### Run file ###########
    experiment.run(num_experiments=num_experiments)
    # experiment.plot_from_saved_results()
if __name__ == "__main__":
    main()