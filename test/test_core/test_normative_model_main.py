
import json

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR


class TestNormativeModel:
    """Test normative model functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()

        # Create template regression model
        self.template_model = BLR.from_args(
            "template", {"alg": "blr", "save_dir": str(self.save_dir)}
        )

        # Create normative model
        self.model = NormativeModel(
            template_regression_model=self.template_model,
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            save_dir=str(self.save_dir),
            inscaler="none",
            outscaler="none",
        )

    def test_model_initialization(self):
        """Test model initialization with valid data."""
        assert self.model.template_regression_model == self.template_model
        assert self.model.save_dir == str(self.save_dir)
        assert self.model.inscaler == "none"
        assert self.model.outscaler == "none"

    def test_model_fit(self):
        """Test model fitting."""
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        # Fit model
        self.model.fit(data)
        assert self.model.is_fitted
        assert len(self.model.regression_models) == 2  # One for each response variable

    def test_model_predict(self):
        """Test model prediction."""
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        self.model.fit(data)

        # Make predictions
        predictions = self.model.predict(data)
        assert "Z" in predictions
        assert predictions.Z.shape == self.data["responses"].shape
        assert not np.isnan(predictions.Z).any()

    def test_model_save_load(self):
        """Test model saving and loading."""
        # Create and fit model
        # Create NormData object
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        self.model.fit(data)

        # Save model
        self.model.save()
        assert (self.save_dir / "model" / "normative_model.json").exists()

        # Load model
        loaded_model = NormativeModel.load(str(self.save_dir))
        assert (
            loaded_model.template_regression_model.__class__
            == self.model.template_regression_model.__class__
        )
        assert loaded_model.is_fitted == self.model.is_fitted

        # Compare predictions
        original_preds = self.model.predict(data)
        loaded_preds = loaded_model.predict(data)
        np.testing.assert_array_almost_equal(original_preds.Z, loaded_preds.Z)

    def test_model_load_after_move(self):
        """A moved model writes its results in the new folder, not the old one."""
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )
        self.model.fit(data)
        assert not list((self.save_dir / "model").glob("*.tmp"))

        moved_dir = self.output_dir / "moved"
        self.save_dir.rename(moved_dir)
        loaded_model = NormativeModel.load(str(moved_dir))
        assert loaded_model.save_dir == str(moved_dir)

        loaded_model.predict(data)
        assert (moved_dir / "results").exists()
        assert not self.save_dir.exists()

    def _fit_and_save(self) -> NormData:
        """Fit and save the model, and return the data it was fitted on."""
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )
        self.model.fit(data)
        self.model.save()
        return data

    def test_model_load_partial(self):
        """A partial load holds only the requested response variables."""
        data = self._fit_and_save()

        loaded = NormativeModel.load(
            str(self.save_dir), response_vars=["response_var_0"]
        )
        assert loaded.response_vars == ["response_var_0"]
        assert list(loaded.regression_models) == ["response_var_0"]
        # Inscalers belong to the covariates, so a partial model keeps them all.
        assert list(loaded.inscalers) == list(self.model.inscalers)

        # A partial model predicts the same values as the full model, when it
        # is given the matching slice of the data, as the runner's chunks are.
        full_preds = self.model.predict(data)
        partial_preds = loaded.predict(data.isel(response_vars=[0]))
        assert partial_preds.Z.shape == (self.data["responses"].shape[0], 1)
        np.testing.assert_allclose(partial_preds.Z.values, full_preds.Z.values[:, [0]])

    def test_model_load_partial_missing_variable(self):
        """A requested variable that is not on disk is named in the error."""
        self._fit_and_save()
        with pytest.raises(ValueError, match="not_in_the_model"):
            NormativeModel.load(str(self.save_dir), response_vars=["not_in_the_model"])

    def test_model_load_partial_renamed_folder(self):
        """A folder whose contents are named differently is an error."""
        self._fit_and_save()
        model_dir = self.save_dir / "model"
        (model_dir / "response_var_0").rename(model_dir / "renamed")
        with pytest.raises(ValueError, match="response_var_0"):
            NormativeModel.load(str(self.save_dir), response_vars=["renamed"])

    def test_model_id_refreshed_by_save(self):
        """Every save writes a new model_id, and load keeps the one on disk."""
        self._fit_and_save()
        with open(
            self.save_dir / "model" / "normative_model.json", encoding="utf-8"
        ) as f:
            on_disk = json.load(f)
        assert on_disk["model_id"] == self.model.model_id

        first_id = self.model.model_id
        self.model.save()
        assert self.model.model_id != first_id
        assert NormativeModel.load(str(self.save_dir)).model_id == self.model.model_id

    def test_model_with_batch_effects(self):
        """Test model with batch effect correction."""
        # Create data with batch effects
        data = NormData.from_ndarrays(
            name="test_data",
            X=self.data["covariates"],
            Y=self.data["responses"],
            batch_effects=self.data["batch_effects"],
            subject_ids=self.data["subject_ids"],
        )

        # Fit and predict
        self.model.fit(data)
        predictions = self.model.predict(data)

        assert "Z" in predictions
        assert predictions.Z.shape == self.data["responses"].shape
        assert not np.isnan(predictions.Z).any()
