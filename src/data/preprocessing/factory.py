"""Factory for creating preprocessing strategy instances."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.preprocessing.base import BasePreprocessor


class PreprocessingStrategy(str, Enum):
    """Available preprocessing strategies."""

    V1_MEDIAN = "v1_median"
    V2_KNN = "v2_knn"
    V3_ITERATIVE = "v3_iterative"
    V4_ROBUST_COL = "v4_robust_col"


class PreprocessorFactory:
    """Factory for creating preprocessor strategy instances.

    Uses a decorator-based registry pattern to register preprocessing strategies.
    Strategies register themselves when imported.

    Example:
        >>> from src.data.preprocessing.factory import PreprocessorFactory
        >>> preprocessor = PreprocessorFactory.create("v1_median")
        >>> preprocessor.fit(X_train)
    """

    _registry: dict[PreprocessingStrategy, type["BasePreprocessor"]] = {}

    @classmethod
    def register(cls, strategy: PreprocessingStrategy):
        """Decorator to register a preprocessing strategy.

        Args:
            strategy: The PreprocessingStrategy enum value to register.

        Returns:
            Decorator function.

        Example:
            >>> @PreprocessorFactory.register(PreprocessingStrategy.V1_MEDIAN)
            ... class V1MedianPreprocessor(BasePreprocessor):
            ...     pass
        """

        def decorator(
            preprocessor_class: type["BasePreprocessor"],
        ) -> type["BasePreprocessor"]:
            cls._registry[strategy] = preprocessor_class
            return preprocessor_class

        return decorator

    @classmethod
    def create(cls, strategy: PreprocessingStrategy | str) -> "BasePreprocessor":
        """Create a preprocessor instance by strategy.

        Args:
            strategy: Either a PreprocessingStrategy enum or string identifier.

        Returns:
            A new instance of the requested preprocessing strategy.

        Raises:
            ValueError: If the strategy is not registered.
        """
        if isinstance(strategy, str):
            try:
                strategy = PreprocessingStrategy(strategy)
            except ValueError:
                available = cls.list_available()
                raise ValueError(
                    f"Unknown preprocessing strategy '{strategy}'. Available: {available}"
                ) from None

        if strategy not in cls._registry:
            available = cls.list_available()
            raise ValueError(f"Strategy '{strategy.value}' not registered. Available: {available}")

        return cls._registry[strategy]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered preprocessing strategies.

        Returns:
            List of strategy identifiers.
        """
        return [s.value for s in cls._registry.keys()]

    @classmethod
    def get_class(cls, strategy: PreprocessingStrategy | str) -> type["BasePreprocessor"]:
        """Get the class for a strategy without instantiating.

        Args:
            strategy: Either a PreprocessingStrategy enum or string identifier.

        Returns:
            The preprocessor strategy class.

        Raises:
            ValueError: If the strategy is not registered.
        """
        if isinstance(strategy, str):
            strategy = PreprocessingStrategy(strategy)

        if strategy not in cls._registry:
            available = cls.list_available()
            raise ValueError(f"Strategy '{strategy.value}' not registered. Available: {available}")

        return cls._registry[strategy]
