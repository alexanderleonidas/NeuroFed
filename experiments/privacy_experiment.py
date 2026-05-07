import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import BaseConfig, AttackConfig
from experiments.experiment_logger import ExperimentLogger
from attacks.attack_model_manager import AttackModelManager
from models.trainable import Trainable
from data.centralised_data import CentralisedDataManager
from training.train_single_model import train_single_model
from training.evaluate_model import evaluate_model


class PrivacyExperiment:
    def __init__(self, shadow_config: BaseConfig, save_model=False, save_results=False, plot_results=False):
        self.save_model = save_model
        self.save_results = save_results
        self.plot_results = plot_results
        if shadow_config is None:
            raise ValueError("No configurations provided for the experiment. Must provide a list of configuration(s).")
        self.attack_config = AttackConfig(shadow_config)

    def run_basic_black_box_attack(self, target_trainable: Trainable, shadow_loaders, run_id:str=None):
        """
        Run a privacy experiment, which includes training shadow models and performing membership inference attacks.
        This queries the global or centralised model to perform the attack. The attacker does not take the role of the client
        or the server in the federated scenario but rather takes the role of an outsider that has black-box access.

        :param shadow_loaders: Data loaders for shadow models, containing member and non-member data.
        :param target_trainable: Target model trainable object
        :type target_trainable: Trainable
        :param run_id: Run ID
        :type run_id: str
        """

        run_id = str(int(time.time())) if run_id is None else run_id
        shadow_member_loader, shadow_non_member_loader = shadow_loaders

        # loaders = CentralisedDataManager(self.attack_config.BATCH_SIZE, self.attack_config.SEED, dataset=self.attack_config.DATASET)
        # shadow_member_loader, shadow_non_member_loader = loaders.get_shadow_loaders(data_size=1.0)

        # Train the shadow model and prepare the attack dataset
        logger = ExperimentLogger(self.attack_config, run_id + '_shadow_model') if self.save_model or self.save_results else None
        attack_manager = AttackModelManager(self.attack_config, self.save_model, self.save_results, self.plot_results)
        attack_train_loader = attack_manager.train_shadow_models(logger, shadow_member_loader, shadow_non_member_loader)

        # Trian the attack model
        logger = ExperimentLogger(self.attack_config, run_id + '_attack_model') if self.save_model or self.save_results else None
        attack_models = attack_manager.create_general_attack_model(logger, attack_train_loader)
        # attack_models = attack_manager.create_class_specific_attack_models(logger, attack_train_loader)

        # Perform the membership inference attack
        # Combine and shuffle the datasets
        all_data = []
        all_labels = []
        true_membership = []

        # Process member data
        for batch in target_trainable.train_loader:
            inputs, labels = batch
            for i in range(inputs.shape[0]):
                all_data.append(inputs[i])
                all_labels.append(labels[i])
                true_membership.append(1)  # 1 for member

        # Process non-member data
        non_member_data = target_trainable.val_loader if target_trainable.val_loader is not None else target_trainable.test_loader
        for batch in non_member_data:
            inputs, labels = batch
            for i in range(inputs.shape[0]):
                all_data.append(inputs[i])
                all_labels.append(labels[i])
                true_membership.append(0)  # 0 for non-member

        # Convert lists to tensors
        all_data = torch.stack(all_data)
        all_labels = torch.tensor(all_labels)

        # Create a random permutation for shuffling
        indices = torch.randperm(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]
        true_membership = [true_membership[i] for i in indices.tolist()]

        # Create a new dataset and loader
        combined_dataset = torch.utils.data.TensorDataset(all_data, all_labels)
        combined_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=target_trainable.config.BATCH_SIZE,
            shuffle=False  # We already shuffled
        )

        # Perform attack on the combined loader
        results, probs = attack_manager.perform_attack(target_trainable.model, combined_loader, attack_models)

        # Calculate accuracy
        attack_accuracy = sum(1 for i, r in enumerate(results) if r == true_membership[i]) / len(results)

        # Calculate separate accuracies for members and non-members
        member_indices = [i for i, m in enumerate(true_membership) if m == 1]
        non_member_indices = [i for i, m in enumerate(true_membership) if m == 0]

        member_success = sum(1 for i in member_indices if results[i] == 1) / len(member_indices)
        non_member_success = sum(1 for i in non_member_indices if results[i] == 0) / len(non_member_indices)

        # Calculate precision and recall
        true_positives = sum(1 for i in member_indices if results[i] == 1)
        false_positives = sum(1 for i in non_member_indices if results[i] == 1)
        false_negatives = sum(1 for i in member_indices if results[i] == 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        print(f"Overall attack accuracy: {attack_accuracy * 100:.2f}%")
        print(f"Attack accuracy on members: {member_success * 100:.2f}%")
        print(f"Attack accuracy on non-members: {non_member_success * 100:.2f}%")
        print(f"Attack Precision: {precision * 100:.2f}%")
        print(f"Attack Recall: {recall * 100:.2f}%")


        if self.plot_results:
            self.plot_roc_curve([probs[i] for i in member_indices], [probs[i] for i in non_member_indices])

    def plot_roc_curve(self, member_probs, non_member_probs):
        y_true = [1] * len(member_probs) + [0] * len(non_member_probs)
        y_scores = member_probs + non_member_probs

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Membership Inference Attack')
        plt.legend(loc="lower right")
        if self.save_results:
            plt.savefig(self.attack_config.RESULTS_PATH + 'mia_roc_curve.png')
        plt.show()

    def plot_from_saved_results(self):
        pass