"""Factory for creating ML model strategy instances."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BaseModel


class ModelType(str, Enum):
    """Available model types."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    XGBOOST = "xgboost"
    LINEAR = "linear"


class ModelFactory:
    """Factory for creating model strategy instances.

    Uses a decorator-based registry pattern to register model strategies.
    Strategies register themselves when imported.

    Example:
        >>> from src.models.factory import ModelFactory, ModelType
        >>> model = ModelFactory.create(ModelType.RANDOM_FOREST)
        >>> model.train(X_train, y_train)
    """

    _registry: dict[ModelType, type["BaseModel"]] = {}

    @classmethod
    def register(cls, model_type: ModelType):
        """Decorator to register a model strategy.

        Args:
            model_type: The ModelType enum value to register.

        Returns:
            Decorator function.

        Example:
            >>> @ModelFactory.register(ModelType.RANDOM_FOREST)
            ... class RandomForestStrategy(BaseModel):
            ...     pass
        """

        def decorator(model_class: type["BaseModel"]) -> type["BaseModel"]:
            cls._registry[model_type] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, model_type: ModelType | str) -> "BaseModel":
        """Create a model instance by type.

        Args:
            model_type: Either a ModelType enum or string identifier.

        Returns:
            A new instance of the requested model strategy.

        Raises:
            ValueError: If the model type is not registered.
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                available = cls.list_available()
                raise ValueError(
                    f"Unknown model type '{model_type}'. Available: {available}"
                ) from None

        if model_type not in cls._registry:
            available = cls.list_available()
            raise ValueError(f"Model '{model_type.value}' not registered. Available: {available}")

        return cls._registry[model_type]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered model types.

        Returns:
            List of model type identifiers.
        """
        return [m.value for m in cls._registry.keys()]

    @classmethod
    def get_class(cls, model_type: ModelType | str) -> type["BaseModel"]:
        """Get the class for a model type without instantiating.

        Args:
            model_type: Either a ModelType enum or string identifier.

        Returns:
            The model strategy class.

        Raises:
            ValueError: If the model type is not registered.
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        if model_type not in cls._registry:
            available = cls.list_available()
            raise ValueError(f"Model '{model_type.value}' not registered. Available: {available}")

        return cls._registry[model_type]
