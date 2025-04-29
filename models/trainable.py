from config import *
from .base_model import FlexibleNet
from .perturbation_optimizer import PerturbationOptimizer
from .feedback_optimizer import DirectFeedbackAlignmentOptimizer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch


class Trainable:
    def __init__(self, config, train_loader=None, val_loader=None, global_fed_model=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self._get_device_type()
        # Create model with the correct name
        self.model = FlexibleNet(config.LAYER_SIZES, self.config.MODEL_TYPE).to(self.device)

        # Setup optimizer and any special configurations
        if not global_fed_model:
            self._setup_model_and_optimizer()

    def _setup_model_and_optimizer(self):
        """Set up the optimizer and any special model configurations"""
        if self.config.MODEL_TYPE == 'BP':
            # Set up a standard backpropagation optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
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
            self.optimizer = DirectFeedbackAlignmentOptimizer(params, self.config.LAYER_SIZES, self.device, lr=self.config.LEARNING_RATE, feedback_noise_scale=scale)
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
        if ModuleValidator.validate(self.model, strict=False):
            self.model = ModuleValidator.fix(self.model)

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm
        )

    def _get_device_type(self):
        # if torch.backends.mps.is_available():
        #     self.device = "mps"
        # elif torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"
        self.device = 'cpu'