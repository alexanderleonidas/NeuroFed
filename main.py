########### Import Experiments ###########
from config import *
from experiments.centralised_performance_comparison import CentralisedPerformanceComparison
from experiments.federated_performance_comparison import FederatedPerformanceComparison

def main():
    ## Choose to save models and results and plot results
    save_model = False
    save_results = False
    plot_results = False # To plot results, run the experiment with save_results = True
    num_experiments = 5

    ## Choose which configurations to run
    configs = [BackpropagationConfig(), DifferentialPrivacyConfig(), DirectFeedbackAlignmentConfig(), PerturbationConfig()]

    ########### Choose Experiment ###########
    experiment = CentralisedPerformanceComparison(configs, save_model, save_results, plot_results)

    # experiment = FederatedPerformanceComparison(configs, save_model, save_results, plot_results)

    ########### Run file ###########
    experiment.run(num_experiments)

if __name__ == "__main__":
    main()