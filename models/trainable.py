from config import *
from .base_model import FlexibleNet
from .logisitc_regression import LogisticRegression
from .perturbation_optimizer import PerturbationOptimizer
from .feedback_optimizer import DirectFeedbackAlignmentOptimizer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch


class Trainable:
    def __init__(self, config, train_loader=None, val_loader=None, test_loader=None, global_fed_model=False, shadow_model=False):
        self._get_device_type()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        # Create a model with the correct name
        if not isinstance(config, AttackConfig) or not shadow_model:
            if config.DATASET in ['mnist', 'fashion']:
                config.LAYER_SIZES.append(10)
            elif config.DATASET == 'emnist':
                config.LAYER_SIZES.append(47)
            else:
                raise NotImplementedError
        self.model = FlexibleNet(config.LAYER_SIZES, self.config.MODEL_TYPE).to(self.device)

        # Setup optimizer and any special configurations
        if not global_fed_model:
            self._setup_model_and_optimizer()

    def _setup_model_and_optimizer(self):
        """Set up the optimizer and any special model configurations"""
        if self.config.MODEL_TYPE == 'BP' or isinstance(self.config, AttackConfig):
            # Set up a standard backpropagation optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=0.0001)
        elif self.config.MODEL_TYPE == 'DP':
            # Set up a differential privacy optimizer and model
            self.privacy_engine = PrivacyEngine()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

            # Determine epochs based on a config type
            if isinstance(self.config, FederatedConfig):
                epochs = self.config.EPOCHS * self.config.COMMUNICATION_ROUNDS
            else:
                epochs = self.config.EPOCHS

            # Get DP parameters from config
            epsilon = getattr(self.config, 'EPSILON', 0.1)
            delta = getattr(self.config, 'DELTA', 0.001)
            max_grad_norm = getattr(self.config, 'MAX_GRAD_NORM', 1.0)

            self._make_model_dp(self.train_loader, epochs, epsilon, delta, max_grad_norm)
        elif self.config.MODEL_TYPE == 'DFA':
            # Get DFA parameters from config
            scale = getattr(self.config, 'FEEDBACK_NOISE_SCALE', 1)
            # Set up Direct Feedback Alignment optimizer
            # params = [param for name, param in self.model.named_parameters() if 'bias' not in name]
            params = self.model.parameters()
            self.optimizer = DirectFeedbackAlignmentOptimizer(params, self.config.LAYER_SIZES, self.device, lr=self.config.LEARNING_RATE, feedback_noise_scale=scale, use_random_noise=True)
        elif self.config.MODEL_TYPE == 'PB':
            """Setup Perturbation-based optimizer"""
            # params = [param for name, param in self.model.named_parameters() if 'bias' not in name]
            params = self.model.parameters()
            sigma = getattr(self.config, 'SIGMA', 0.000001)
            self.optimizer = PerturbationOptimizer(params, lr=self.config.LEARNING_RATE, sigma=sigma)
        else:
            raise ValueError(f"Invalid model type")


    def _make_model_dp(self, train_loader, epochs, epsilon, delta, max_grad_norm):
        """Apply differential privacy to the model"""
        if ModuleValidator.validate(self.model, strict=True):
            self.model = ModuleValidator.fix(self.model)

        # Create a wide range of alphas, especially for low epsilon values
        base_alphas = [1 + x / 10.0 for x in range(1, 100)]
        medium_alphas = list(range(12, 100))
        high_alphas = list(range(100, int(1e6), 10))  # Add much higher values for low epsilon


        if epsilon <= 0.5:
            alphas = base_alphas + medium_alphas + high_alphas
        elif epsilon <= 1.0:
            alphas = base_alphas + medium_alphas + list(range(100, 300, 10))
        else:
            alphas = base_alphas + medium_alphas

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
            alphas=high_alphas
        )

    def _get_device_type(self):
        # if torch.backends.mps.is_available():
        #     self.device = "mps"
        # elif torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"
        self.device = 'cpu'