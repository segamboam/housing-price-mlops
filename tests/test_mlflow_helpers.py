"""Tests for MLflow helper functions."""

from unittest.mock import MagicMock, patch

from src.utils.mlflow_helpers import initialize_mlflow, tag_model_version


class TestInitializeMlflow:
    """Tests for initialize_mlflow function."""

    @patch("src.utils.mlflow_helpers.mlflow")
    @patch("src.utils.mlflow_helpers.MlflowClient")
    @patch("src.utils.mlflow_helpers.get_settings")
    def test_returns_mlflow_client(self, mock_settings, mock_client_class, mock_mlflow):
        """Returns an MlflowClient instance."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.mlflow_tracking_uri = "http://localhost:5000"
        mock_settings.return_value = mock_settings_instance

        result = initialize_mlflow()

        assert mock_client_class.called
        assert result == mock_client_class.return_value

    @patch("src.utils.mlflow_helpers.mlflow")
    @patch("src.utils.mlflow_helpers.MlflowClient")
    @patch("src.utils.mlflow_helpers.get_settings")
    def test_sets_tracking_uri(self, mock_settings, mock_client_class, mock_mlflow):
        """Sets MLflow tracking URI."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.mlflow_tracking_uri = "http://test:5000"
        mock_settings.return_value = mock_settings_instance

        initialize_mlflow()

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://test:5000")

    @patch("src.utils.mlflow_helpers.mlflow")
    @patch("src.utils.mlflow_helpers.MlflowClient")
    @patch("src.utils.mlflow_helpers.get_settings")
    def test_uses_custom_tracking_uri(self, mock_settings, mock_client_class, mock_mlflow):
        """Uses custom tracking URI when provided."""
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance

        initialize_mlflow(tracking_uri="http://custom:8000")

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://custom:8000")

    @patch("src.utils.mlflow_helpers.mlflow")
    @patch("src.utils.mlflow_helpers.MlflowClient")
    @patch("src.utils.mlflow_helpers.get_settings")
    def test_configures_s3_by_default(self, mock_settings, mock_client_class, mock_mlflow):
        """Configures S3 credentials by default."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.mlflow_tracking_uri = "http://localhost:5000"
        mock_settings.return_value = mock_settings_instance

        initialize_mlflow()

        mock_settings_instance.configure_mlflow_s3.assert_called_once()

    @patch("src.utils.mlflow_helpers.mlflow")
    @patch("src.utils.mlflow_helpers.MlflowClient")
    @patch("src.utils.mlflow_helpers.get_settings")
    def test_skips_s3_config_when_disabled(self, mock_settings, mock_client_class, mock_mlflow):
        """Skips S3 configuration when disabled."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.mlflow_tracking_uri = "http://localhost:5000"
        mock_settings.return_value = mock_settings_instance

        initialize_mlflow(configure_s3=False)

        mock_settings_instance.configure_mlflow_s3.assert_not_called()


class TestTagModelVersion:
    """Tests for tag_model_version function."""

    def test_updates_model_description(self):
        """Updates model version description."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="1",
            model_type="gradient_boost",
            preprocessing="v1_median",
            test_r2=0.85,
            test_rmse=3.5,
        )

        mock_client.update_model_version.assert_called_once()
        call_args = mock_client.update_model_version.call_args
        assert call_args.kwargs["name"] == "test_model"
        assert call_args.kwargs["version"] == "1"
        assert "gradient_boost" in call_args.kwargs["description"]
        assert "v1_median" in call_args.kwargs["description"]

    def test_sets_model_type_tag(self):
        """Sets model_type tag."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="2",
            model_type="random_forest",
            preprocessing="v2_knn",
            test_r2=0.9,
            test_rmse=2.0,
        )

        # Check that set_model_version_tag was called with model_type
        calls = mock_client.set_model_version_tag.call_args_list
        model_type_calls = [c for c in calls if c.args[2] == "model_type"]
        assert len(model_type_calls) == 1
        assert model_type_calls[0].args[3] == "random_forest"

    def test_sets_preprocessing_tag(self):
        """Sets preprocessing tag."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="1",
            model_type="gradient_boost",
            preprocessing="v3_iterative",
            test_r2=0.88,
            test_rmse=3.0,
        )

        calls = mock_client.set_model_version_tag.call_args_list
        preprocessing_calls = [c for c in calls if c.args[2] == "preprocessing"]
        assert len(preprocessing_calls) == 1
        assert preprocessing_calls[0].args[3] == "v3_iterative"

    def test_sets_source_tag(self):
        """Sets source tag with default value."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="1",
            model_type="gradient_boost",
            preprocessing="v1_median",
            test_r2=0.85,
            test_rmse=3.5,
        )

        calls = mock_client.set_model_version_tag.call_args_list
        source_calls = [c for c in calls if c.args[2] == "source"]
        assert len(source_calls) == 1
        assert source_calls[0].args[3] == "unknown"

    def test_sets_custom_source_tag(self):
        """Sets custom source tag when provided."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="1",
            model_type="gradient_boost",
            preprocessing="v1_median",
            test_r2=0.85,
            test_rmse=3.5,
            source="experiment_runner",
        )

        calls = mock_client.set_model_version_tag.call_args_list
        source_calls = [c for c in calls if c.args[2] == "source"]
        assert source_calls[0].args[3] == "experiment_runner"

    def test_formats_r2_in_description(self):
        """Formats R2 score correctly in description."""
        mock_client = MagicMock()

        tag_model_version(
            client=mock_client,
            model_name="test_model",
            version="1",
            model_type="gradient_boost",
            preprocessing="v1_median",
            test_r2=0.8567,
            test_rmse=3.456,
        )

        description = mock_client.update_model_version.call_args.kwargs["description"]
        assert "0.8567" in description
        assert "3.4560" in description
