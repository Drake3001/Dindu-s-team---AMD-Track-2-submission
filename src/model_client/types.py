class ModelRequestError(RuntimeError):
    """Raised when a model API request fails."""


class ModelResponseError(RuntimeError):
    """Raised when a model API response has an unexpected shape."""
