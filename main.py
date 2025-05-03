########### Import Experiments ###########
from config import *
from experiments.centralised_performance_comparison import CentralisedPerformanceComparison
from experiments.federated_performance_comparison import FederatedPerformanceComparison

def main():
    ## Choose to save models and results and plot results
    save_model = True
    save_results = True
    plot_results = True # To plot results, run the experiment with save_results = True
    num_experiments = 1

    ## Choose which configurations to run
    configs = [BackpropagationConfig(), DirectFeedbackAlignmentConfig(), PerturbationConfig(), DifferentialPrivacyConfig()]
    # configs = [DirectFeedbackAlignmentConfig(), PerturbationConfig()]
    ########### Choose Experiment ###########
    experiment = CentralisedPerformanceComparison(configs, save_model, save_results, plot_results)

    # experiment = FederatedPerformanceComparison(configs, save_model, save_results, plot_results)

    ########### Run file ###########
    # experiment.run(num_experiments)
    experiment.plot_from_saved_results2()
if __name__ == "__main__":
    main()