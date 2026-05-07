from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    LAYER_SIZES: list[int] = field(default_factory=lambda: [784, 100, 100, 100]) # Output layer size is determined by dataset
    EPOCHS: int = 20
    BATCH_SIZE: int = 128
    VERBOSE: bool = True
    SEED: int = 42
    DATASET: str = "mnist"
    RESULTS_PATH: str = "./results/"
    SAVED_MODELS_PATH: str = "./saved_models/"
    LEARNING_RATE: float = None
    NAME: str = None
    MODEL_TYPE: str = None


@dataclass
class BackpropagationConfig(BaseConfig):
    LEARNING_RATE: float = 0.001
    NAME: str = "Backpropagation"
    MODEL_TYPE: str = "BP"


@dataclass
class DifferentialPrivacyConfig(BaseConfig):
    LEARNING_RATE: float = 0.001
    EPSILON: float = 5
    DELTA: float = 1e-5
    MAX_GRAD_NORM: float = 1.0
    NAME: str = "Differential Privacy"
    MODEL_TYPE: str = "DP"


@dataclass
class PerturbationConfig(BaseConfig):
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 6.81e-4
    SIGMA: float = 1e-4
    NAME: str = "Weight Perturbation"
    MODEL_TYPE: str = "PB"


@dataclass
class DirectFeedbackAlignmentConfig(BaseConfig):
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1/2048
    FEEDBACK_NOISE_SCALE: float = 1
    NAME: str = "Direct Feedback Alignment"
    MODEL_TYPE: str = "DFA"

@dataclass
class FederatedConfig(BaseConfig):
    NUM_CLIENTS: int = 3
    CLIENT_FRACTION: float = 0.8
    COMMUNICATION_ROUNDS: int = 10
    IID: bool = True
    EXPERIMENT_TYPE: str = 'Federated'

    def __init__(self, model_config: BaseConfig):
        if not isinstance(model_config, BaseConfig):
            raise TypeError("model_config must be an instance of BaseConfig")

        # Copy base config values
        base_attrs = asdict(model_config)
        for key, value in base_attrs.items():
            setattr(self, key, value)

        self.NAME = f"Federated {model_config.NAME}"
        self.EPOCHS = 1  # Each client trains for 'n' epochs per communication round

@dataclass
class AttackConfig(BaseConfig):
    NAME: str = 'Attack Model'
    ATTACK_MODEL_TYPE: str = 'nn'
    ATTACK_TYPE: str = 'basic'
    EXPERIMENT_TYPE: str = "Attack"
    TOP_K_FEATURES: int = 3
    LAYER_SIZES = [TOP_K_FEATURES, 100, 100, 1] # Only used for 'nn' ATTACK_MODEL_TYPE
    EPOCHS: int = 20
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.001
    NUM_SHADOW_MODELS: int = 1

    def __init__(self, shadow_model_config: BaseConfig):
        if not isinstance(shadow_model_config, BaseConfig):
            raise TypeError("model_config must be an instance of BaseConfig")
        self.SHADOW_MODEL_CONFIG = shadow_model_config
        # # Copy base config values
        # base_attrs = asdict(shadow_model_config)
        # for key, value in base_attrs.items():
        #     setattr(self, key, value)
        #
        # self.NAME = f"Attack {shadow_model_config.NAME}"