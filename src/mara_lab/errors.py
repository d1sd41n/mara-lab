class MaraLabError(RuntimeError):
    """Base error for expected, user-actionable failures."""


class ConfigurationError(MaraLabError):
    """Raised when an experiment configuration cannot be loaded."""


class BackendUnavailableError(MaraLabError):
    """Raised when a requested generation backend cannot run."""


class MemoryBudgetExceededError(MaraLabError):
    """Raised when a generation exceeds the configured CUDA memory ceiling."""


class PromptTooLongError(MaraLabError):
    """Raised when a prompt would be truncated by a model tokenizer."""


class ReproductionMismatchError(MaraLabError):
    """Raised when a pinned-host reproduction is not byte-identical."""


class ModelMaterializationError(MaraLabError):
    """Raised when a local model snapshot cannot be prepared safely."""


class TrainingError(MaraLabError):
    """Raised when a character adapter cannot be prepared or trained safely."""
