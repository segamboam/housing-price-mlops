"""Traffic router for champion/challenger model serving.

Implements blue-green (A/B) traffic splitting between two model bundles.
The router selects which model serves each prediction request based on
a configurable weight.
"""

import random

from src.artifacts.bundle import MLArtifactBundle


class TrafficRouter:
    """Routes prediction traffic between champion and challenger models.

    If only the champion is loaded, all traffic goes to it regardless of weight.
    If both are loaded, traffic is split according to ``champion_weight``.

    Args:
        champion: The champion (primary) artifact bundle.
        challenger: The challenger (candidate) artifact bundle, or None.
        champion_weight: Probability of routing to champion (0.0 - 1.0).
    """

    def __init__(
        self,
        champion: MLArtifactBundle | None,
        challenger: MLArtifactBundle | None,
        champion_weight: float = 0.5,
    ) -> None:
        self.champion = champion
        self.challenger = challenger
        self.champion_weight = champion_weight

    def select(self) -> tuple[MLArtifactBundle, str]:
        """Select which model bundle should serve the current request.

        Returns:
            Tuple of (selected_bundle, alias_name).

        Raises:
            RuntimeError: If no model is available (both champion and
                challenger are None).
        """
        if self.champion is None and self.challenger is None:
            raise RuntimeError("No model available for predictions")

        # If only one model is loaded, use it
        if self.challenger is None:
            return self.champion, "champion"  # type: ignore[return-value]
        if self.champion is None:
            return self.challenger, "challenger"

        # Both loaded: split by weight
        if random.random() < self.champion_weight:
            return self.champion, "champion"
        return self.challenger, "challenger"

    @property
    def effective_split(self) -> dict[str, float]:
        """Return the effective traffic split considering loaded models.

        If only one model is loaded, it gets 100% of traffic regardless
        of the configured weight.
        """
        if self.champion is None and self.challenger is None:
            return {"champion": 0.0, "challenger": 0.0}
        if self.challenger is None:
            return {"champion": 1.0, "challenger": 0.0}
        if self.champion is None:
            return {"champion": 0.0, "challenger": 1.0}
        return {
            "champion": self.champion_weight,
            "challenger": round(1.0 - self.champion_weight, 4),
        }
