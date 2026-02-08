"""Property-based tests for the prediction API using Hypothesis.

Validates model consumption robustness by generating thousands of random
inputs within the valid domain and verifying the API always behaves correctly.

This catches edge cases that hand-written examples miss:
- Boundary values (e.g. CRIM=0, RM=15, DIS=0.001)
- Unusual but valid combinations of features
- Floating-point precision edge cases
"""

import math

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.api.schemas import HousingFeatures

# ---------------------------------------------------------------------------
# Hypothesis strategy: generate valid HousingFeatures dicts
# ---------------------------------------------------------------------------


@st.composite
def valid_housing_features(draw: st.DrawFn) -> dict:
    """Generate a random but valid HousingFeatures payload.

    Respects every Pydantic constraint defined in HousingFeatures:
    - CRIM: float >= 0            (capped at 100 for realism)
    - ZN: float [0, 100]
    - INDUS: float [0, 100]
    - CHAS: int {0, 1}
    - NOX: float [0, 1]
    - RM: float [1, 15]
    - AGE: float [0, 100]
    - DIS: float > 0              (min 0.001 to avoid precision issues)
    - RAD: int [1, 24]
    - TAX: float >= 0             (capped at 1000)
    - PTRATIO: float >= 0         (capped at 30)
    - B: float [0, 400]
    - LSTAT: float [0, 100]
    """
    return {
        "CRIM": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "ZN": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "INDUS": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "CHAS": draw(st.sampled_from([0, 1])),
        "NOX": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "RM": draw(st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False)),
        "AGE": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "DIS": draw(
            st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "RAD": draw(st.integers(min_value=1, max_value=24)),
        "TAX": draw(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        "PTRATIO": draw(
            st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False)
        ),
        "B": draw(st.floats(min_value=0.0, max_value=400.0, allow_nan=False, allow_infinity=False)),
        "LSTAT": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
    }


# A baseline valid payload used to mutate for invalid-input tests
_VALID_BASELINE = {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPropertyBasedPrediction:
    """Property-based tests: any valid input produces a correct prediction."""

    @given(features=valid_housing_features())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_input_always_produces_finite_prediction(
        self, features: dict, mock_artifact_bundle
    ):
        """Any valid HousingFeatures input returns 200 with a finite prediction.

        Properties verified:
        - HTTP 200
        - prediction is a finite float (not NaN, not Inf)
        - served_by is present and non-empty
        - model_version is present
        """
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict", json=features)

        assert (
            response.status_code == 200
        ), f"Expected 200 for valid input, got {response.status_code}: {response.text}"

        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert math.isfinite(data["prediction"]), f"Prediction is not finite: {data['prediction']}"
        assert data.get("served_by") in ("champion", "challenger")
        assert data.get("model_version")

    @given(features=valid_housing_features())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_single_vs_batch_consistency(self, features: dict, mock_artifact_bundle):
        """Same input via /predict and /predict/batch gives the same prediction value.

        This ensures batch processing doesn't introduce drift or rounding
        differences compared to single predictions.
        """
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            single_resp = client.post("/predict", json=features)
            batch_resp = client.post("/predict/batch", json={"items": [features]})

        assert single_resp.status_code == 200
        assert batch_resp.status_code == 200

        single_pred = single_resp.json()["prediction"]
        batch_pred = batch_resp.json()["predictions"][0]["prediction"]

        assert (
            single_pred == batch_pred
        ), f"Single ({single_pred}) != Batch ({batch_pred}) for same input"

    @given(features=valid_housing_features())
    @settings(max_examples=200, deadline=None)
    def test_schema_roundtrip(self, features: dict):
        """Any generated valid features survive serialization roundtrip.

        HousingFeatures(**dict) -> .model_dump() -> HousingFeatures(**dict)
        should not lose or corrupt data.
        """
        model = HousingFeatures(**features)
        dumped = model.model_dump()
        reconstructed = HousingFeatures(**dumped)

        assert (
            model == reconstructed
        ), f"Roundtrip mismatch: {model.model_dump()} != {reconstructed.model_dump()}"

    @given(
        field_to_break=st.sampled_from(list(_VALID_BASELINE.keys())),
        bad_value=st.sampled_from(
            [
                "not_a_number",  # wrong type
                None,  # null
                -999.0,  # negative (invalid for most fields)
            ]
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_input_always_returns_422(
        self, field_to_break: str, bad_value, mock_artifact_bundle
    ):
        """Mutating a single field to an invalid value always returns 422.

        Verifies the API consistently rejects malformed inputs with
        the unified ErrorResponse schema.
        """
        payload = {**_VALID_BASELINE, field_to_break: bad_value}

        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict", json=payload)

        assert response.status_code == 422, (
            f"Expected 422 for broken field '{field_to_break}={bad_value}', "
            f"got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "detail" in data
        assert "code" in data
        assert data["code"] == "VALIDATION_ERROR"


class TestBoundaryValues:
    """Explicit boundary tests for extreme valid values.

    Hypothesis finds boundaries naturally, but these tests serve as
    a regression safety net for known edge cases.
    """

    BOUNDARY_CASES = [
        {
            "label": "all_minimums",
            "overrides": {
                "CRIM": 0.0,
                "ZN": 0.0,
                "INDUS": 0.0,
                "CHAS": 0,
                "NOX": 0.0,
                "RM": 1.0,
                "AGE": 0.0,
                "DIS": 0.001,
                "RAD": 1,
                "TAX": 0.0,
                "PTRATIO": 0.0,
                "B": 0.0,
                "LSTAT": 0.0,
            },
        },
        {
            "label": "all_maximums",
            "overrides": {
                "CRIM": 100.0,
                "ZN": 100.0,
                "INDUS": 100.0,
                "CHAS": 1,
                "NOX": 1.0,
                "RM": 15.0,
                "AGE": 100.0,
                "DIS": 100.0,
                "RAD": 24,
                "TAX": 1000.0,
                "PTRATIO": 30.0,
                "B": 400.0,
                "LSTAT": 100.0,
            },
        },
        {"label": "chas_boundary_0", "overrides": {"CHAS": 0}},
        {"label": "chas_boundary_1", "overrides": {"CHAS": 1}},
        {"label": "dis_near_zero", "overrides": {"DIS": 0.001}},
        {"label": "rm_minimum", "overrides": {"RM": 1.0}},
        {"label": "rm_maximum", "overrides": {"RM": 15.0}},
        {"label": "rad_minimum", "overrides": {"RAD": 1}},
        {"label": "rad_maximum", "overrides": {"RAD": 24}},
    ]

    @staticmethod
    def _make_payload(overrides: dict) -> dict:
        return {**_VALID_BASELINE, **overrides}

    def test_boundary_values_dont_crash(self, mock_artifact_bundle):
        """All boundary combinations return 200 with a finite prediction."""
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            for case in self.BOUNDARY_CASES:
                payload = self._make_payload(case["overrides"])
                response = client.post("/predict", json=payload)

                assert response.status_code == 200, (
                    f"Boundary '{case['label']}' failed with {response.status_code}: "
                    f"{response.text}"
                )

                pred = response.json()["prediction"]
                assert math.isfinite(
                    pred
                ), f"Boundary '{case['label']}' gave non-finite prediction: {pred}"

    def test_missing_field_returns_422(self, mock_artifact_bundle):
        """Removing any single required field returns 422."""
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            for field in _VALID_BASELINE:
                payload = {k: v for k, v in _VALID_BASELINE.items() if k != field}
                response = client.post("/predict", json=payload)

                assert (
                    response.status_code == 422
                ), f"Missing '{field}' should give 422, got {response.status_code}"
