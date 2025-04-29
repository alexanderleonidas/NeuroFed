from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    LAYER_SIZES: list[int] = field(default_factory=lambda: [784, 512, 216, 10])
    EPOCHS: int = 200
    BATCH_SIZE: int = 128
    VERBOSE: bool = True
    SEED: int = 42
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
    BATCH_SIZE: int = 1000
    LEARNING_RATE: float = 0.0001
    SIGMA: float = 1e-6
    NAME: str = "Weight Perturbation"
    MODEL_TYPE: str = "PB"


@dataclass
class DirectFeedbackAlignmentConfig(BaseConfig):
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.001
    FEEDBACK_NOISE_SCALE: float = 1
    NAME: str = "Direct Feedback Alignment"
    MODEL_TYPE: str = "DFA"


@dataclass
class FederatedConfig(BaseConfig):
    NUM_CLIENTS: int = 3
    CLIENT_FRACTION: float = 1
    COMMUNICATION_ROUNDS: int = 2
    IID: bool = False
    EXPERIMENT_TYPE: str = 'Federated'

    def __init__(self, model_config: BaseConfig):
        if not isinstance(model_config, BaseConfig):
            raise TypeError("model_config must be an instance of BaseConfig")

        # Copy base config values
        base_attrs = asdict(model_config)
        for key, value in base_attrs.items():
            setattr(self, key, value)

        self.NAME = f"Federated {model_config.NAME}"