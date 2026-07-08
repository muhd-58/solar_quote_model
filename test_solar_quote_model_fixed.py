"""
Comprehensive Unit Tests for Solar Quote Model (Fixed Version)
===============================================================

Tests cover:
  - Input validation and error handling
  - Model training and persistence
  - Design system generation with edge cases
  - Quantile prediction fallback mechanisms
  - Guardrail enforcement (roof area, battery, inverter snapping)
  - Feature importance extraction
"""

import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Import the model class
from solar_quote_model_fixed import (
    SolarQuoteModelFixed,
    TARGETS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    AVAILABLE_INVERTER_SIZES_KW,
    REGIONAL_IRRADIANCE,
)

# Suppress model training logs during tests
logging.getLogger("solar_quote_model_fixed").setLevel(logging.ERROR)


class TestModelInitialization(unittest.TestCase):
    """Test model initialization and basic setup."""

    def test_init_default_panel_wattage(self):
        """Model initializes with default panel wattage."""
        model = SolarQuoteModelFixed()
        self.assertEqual(model.panel_wattage, 620)

    def test_init_custom_panel_wattage(self):
        """Model initializes with custom panel wattage."""
        model = SolarQuoteModelFixed(panel_wattage=700)
        self.assertEqual(model.panel_wattage, 700)

    def test_empty_model_containers(self):
        """Model initializes with empty model containers."""
        model = SolarQuoteModelFixed()
        self.assertEqual(len(model.models), 0)
        self.assertEqual(len(model.quantile_models), 0)
        self.assertEqual(len(model.metrics), 0)


class TestDataLoading(unittest.TestCase):
    """Test historical data loading with validation."""

    def setUp(self):
        """Create test data directory and model."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = SolarQuoteModelFixed()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def _create_test_csv(self, filename, n_rows=50, include_nan=False):
        """Helper to create test CSV."""
        np.random.seed(42)
        data = {
            "monthly_kwh_demand": np.random.uniform(200, 2000, n_rows),
            "peak_demand_kw": np.random.uniform(3, 15, n_rows),
            "region": np.random.choice(["north", "south", "east"], n_rows),
            "roof_area_m2": np.random.uniform(20, 150, n_rows),
            "battery_requested": np.random.randint(0, 2, n_rows),
            "battery_kwh": np.random.uniform(0, 20, n_rows),
            "system_kw": np.random.uniform(3, 20, n_rows),
            "panel_wattage": 620,
            "inverter_kw": np.random.uniform(5, 30, n_rows),
            "final_price": np.random.uniform(10000, 50000, n_rows),
        }

        if include_nan:
            data["roof_area_m2"][0] = np.nan
            data["battery_kwh"][1] = np.nan

        df = pd.DataFrame(data)
        path = Path(self.temp_dir) / filename
        df.to_csv(path, index=False)
        return path

    def test_load_valid_csv(self):
        """Successfully load valid CSV with all required columns."""
        path = self._create_test_csv("valid.csv")
        df = self.model.load_historical_data(str(path), drop_na=False)
        self.assertEqual(len(df), 50)
        self.assertIn("system_kw", df.columns)

    def test_load_nonexistent_file(self):
        """Raise error when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.model.load_historical_data("/nonexistent/file.csv")

    def test_load_missing_required_columns(self):
        """Raise error when required columns are missing."""
        path = Path(self.temp_dir) / "incomplete.csv"
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        df.to_csv(path, index=False)

        with self.assertRaises(ValueError) as ctx:
            self.model.load_historical_data(str(path))
        self.assertIn("Missing columns", str(ctx.exception))

    def test_load_with_nan_imputation(self):
        """NaN values in roof_area_m2 are imputed with median."""
        path = self._create_test_csv("with_nan.csv", include_nan=True)
        df = self.model.load_historical_data(str(path), drop_na=False)
        # Should have NaN imputed
        self.assertFalse(df["roof_area_m2"].isna().any())

    def test_load_drop_na_removes_rows(self):
        """drop_na=True removes rows with missing values."""
        path = self._create_test_csv("with_nan.csv", n_rows=50, include_nan=True)
        df = self.model.load_historical_data(str(path), drop_na=True)
        # Some rows should be dropped
        self.assertLess(len(df), 50)


class TestInputValidation(unittest.TestCase):
    """Test input validation methods."""

    def setUp(self):
        self.model = SolarQuoteModelFixed()

    def test_validate_input_valid_value(self):
        """Valid positive value passes validation."""
        result = self.model._validate_input(100.5, "test_param")
        self.assertEqual(result, 100.5)

    def test_validate_input_nan(self):
        """NaN raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.model._validate_input(np.nan, "test_param")
        self.assertIn("NaN", str(ctx.exception))

    def test_validate_input_inf(self):
        """Infinity raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.model._validate_input(np.inf, "test_param")
        self.assertIn("infinite", str(ctx.exception))

    def test_validate_input_below_minimum(self):
        """Value below minimum is clamped."""
        result = self.model._validate_input(-10, "test_param", min_val=0)
        self.assertEqual(result, 0)

    def test_validate_input_at_minimum(self):
        """Value at minimum boundary passes."""
        result = self.model._validate_input(0, "test_param", min_val=0)
        self.assertEqual(result, 0)


class TestInverterSnapping(unittest.TestCase):
    """Test inverter size snapping to available sizes."""

    def setUp(self):
        self.model = SolarQuoteModelFixed()

    def test_snap_exact_available_size(self):
        """Exact match to available size returns unchanged."""
        result = self.model.snap_to_valid_inverter(10.0)
        self.assertEqual(result, 10.0)
        self.assertIn(result, AVAILABLE_INVERTER_SIZES_KW)

    def test_snap_between_available_sizes(self):
        """Value between sizes snaps to nearest."""
        result = self.model.snap_to_valid_inverter(6.0)
        self.assertIn(result, AVAILABLE_INVERTER_SIZES_KW)
        self.assertTrue(result in [5, 8])  # Closest sizes

    def test_snap_below_minimum(self):
        """Small value snaps to minimum available."""
        result = self.model.snap_to_valid_inverter(0.5)
        self.assertEqual(result, min(AVAILABLE_INVERTER_SIZES_KW))

    def test_snap_above_maximum(self):
        """Large value snaps to maximum available."""
        result = self.model.snap_to_valid_inverter(100.0)
        self.assertEqual(result, max(AVAILABLE_INVERTER_SIZES_KW))

    def test_snap_invalid_input(self):
        """NaN or inf input handled gracefully."""
        result = self.model.snap_to_valid_inverter(np.nan)
        self.assertIn(result, AVAILABLE_INVERTER_SIZES_KW)

        result = self.model.snap_to_valid_inverter(np.inf)
        self.assertIn(result, AVAILABLE_INVERTER_SIZES_KW)


class TestRoofAreaValidation(unittest.TestCase):
    """Test roof area validation."""

    def setUp(self):
        self.model = SolarQuoteModelFixed()

    def test_roof_sufficient_for_system(self):
        """System that fits in roof passes validation."""
        is_valid, msg = self.model.validate_roof_area(100, 10)
        self.assertTrue(is_valid)
        self.assertIn("sufficient", msg.lower())

    def test_roof_insufficient_for_system(self):
        """System too large for roof fails validation."""
        is_valid, msg = self.model.validate_roof_area(20, 10)
        self.assertFalse(is_valid)
        self.assertIn("available", msg.lower())

    def test_roof_area_not_specified(self):
        """When roof_area >= 999, treated as unknown."""
        is_valid, msg = self.model.validate_roof_area(999, 50)
        self.assertTrue(is_valid)
        self.assertIn("not specified", msg.lower())

    def test_roof_area_negative(self):
        """Negative roof area clamped to 0."""
        is_valid, msg = self.model.validate_roof_area(-50, 10)
        # Should fail because 0 m² can't fit system
        self.assertFalse(is_valid)

    def test_roof_zero_system(self):
        """Zero system size passes (edge case)."""
        is_valid, msg = self.model.validate_roof_area(50, 0)
        self.assertTrue(is_valid)


class TestBatteryImpactCalculation(unittest.TestCase):
    """Test battery system impact on inverter sizing."""

    def setUp(self):
        self.model = SolarQuoteModelFixed()

    def test_no_battery_requested(self):
        """When battery_requested=0, inverter unchanged."""
        adjusted, note = self.model.calculate_battery_impact(0, 0, 10.0)
        self.assertEqual(adjusted, 10.0)
        self.assertIn("No battery", note)

    def test_battery_with_zero_capacity(self):
        """Battery requested but capacity=0 returns unchanged inverter."""
        adjusted, note = self.model.calculate_battery_impact(1, 0, 10.0)
        self.assertEqual(adjusted, 10.0)
        self.assertIn("not specified", note.lower())

    def test_battery_bumps_inverter_size(self):
        """Large battery capacity increases inverter size."""
        adjusted, note = self.model.calculate_battery_impact(1, 20, 5.0)
        self.assertGreater(adjusted, 5.0)
        self.assertIn("bumped", note.lower())

    def test_battery_small_no_change(self):
        """Small battery with large inverter doesn't change inverter."""
        adjusted, note = self.model.calculate_battery_impact(1, 5, 30.0)
        self.assertEqual(adjusted, 30.0)

    def test_battery_invalid_capacity(self):
        """Negative battery capacity handled gracefully."""
        adjusted, note = self.model.calculate_battery_impact(1, -10, 10.0)
        self.assertEqual(adjusted, 10.0)


class TestModelTraining(unittest.TestCase):
    """Test model training pipeline."""

    def setUp(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n = 100
        self.model = SolarQuoteModelFixed()

        demand = np.random.uniform(200, 2000, n)
        self.df = pd.DataFrame(
            {
                "monthly_kwh_demand": demand,
                "peak_demand_kw": demand / 30 / 4,
                "region": np.random.choice(["north", "south"], n),
                "roof_area_m2": np.random.uniform(20, 150, n),
                "battery_requested": np.random.randint(0, 2, n),
                "battery_kwh": np.random.uniform(0, 20, n),
                "system_kw": demand * 0.012 + np.random.normal(0, 0.3, n),
                "panel_wattage": 620,
                "inverter_kw": demand * 0.012 * 1.05 + np.random.normal(0, 0.2, n),
                "final_price": demand * 4 + np.random.normal(0, 100, n),
            }
        )
        self.df["system_kw"] = np.maximum(self.df["system_kw"], 0.5)
        self.df["inverter_kw"] = np.maximum(self.df["inverter_kw"], 2)
        self.df["final_price"] = np.maximum(self.df["final_price"], 5000)

    def test_train_models_returns_fitted_models(self):
        """Training produces Pipeline objects for each target."""
        models, metrics = self.model.train_models(self.df, cv_folds=3)
        self.model.models = models

        self.assertEqual(len(models), len(TARGETS))
        for target in TARGETS:
            self.assertIn(target, models)
            self.assertTrue(hasattr(models[target], "predict"))

    def test_train_models_produces_metrics(self):
        """Training produces evaluation metrics."""
        models, metrics = self.model.train_models(self.df, cv_folds=3)
        self.model.models = models

        self.assertEqual(len(metrics), len(TARGETS))
        for target in TARGETS:
            self.assertIn("mae_test", metrics[target])
            self.assertIn("r2_test", metrics[target])
            self.assertIn("cv_mae_mean", metrics[target])
            self.assertGreater(metrics[target]["cv_mae_mean"], 0)

    def test_train_models_creates_quantile_models(self):
        """Training creates quantile models for uncertainty."""
        models, metrics = self.model.train_models(self.df, cv_folds=3, quantiles=[0.1, 0.9])
        self.model.models = models

        # Check quantile models exist
        for target in TARGETS:
            for q in [0.1, 0.9]:
                key = f"{target}_q{q}"
                self.assertIn(key, self.model.quantile_models)

    def test_train_insufficient_data(self):
        """Training with too few samples raises error."""
        small_df = self.df.head(3)
        with self.assertRaises(ValueError):
            self.model.train_models(small_df)

    def test_feature_importance_extraction(self):
        """Feature importance can be extracted after training."""
        models, metrics = self.model.train_models(self.df, cv_folds=3)
        self.model.models = models

        for target in TARGETS:
            imp_df = self.model.get_feature_importance(target, top_n=3)
            self.assertEqual(len(imp_df), 3)
            self.assertIn("feature", imp_df.columns)
            self.assertIn("importance", imp_df.columns)

    def test_feature_importance_nonexistent_target(self):
        """Requesting importance for untrained target raises error."""
        with self.assertRaises(ValueError):
            self.model.get_feature_importance("nonexistent_target")


class TestModelPersistence(unittest.TestCase):
    """Test saving and loading trained models."""

    def setUp(self):
        """Train a model and prepare for save/load tests."""
        np.random.seed(42)
        n = 50
        self.model = SolarQuoteModelFixed(panel_wattage=700)
        self.temp_dir = tempfile.mkdtemp()

        demand = np.random.uniform(200, 2000, n)
        df = pd.DataFrame(
            {
                "monthly_kwh_demand": demand,
                "peak_demand_kw": demand / 30 / 4,
                "region": np.random.choice(["north", "south"], n),
                "roof_area_m2": np.random.uniform(20, 150, n),
                "battery_requested": np.random.randint(0, 2, n),
                "battery_kwh": np.random.uniform(0, 20, n),
                "system_kw": demand * 0.012 + np.random.normal(0, 0.3, n),
                "panel_wattage": 620,
                "inverter_kw": demand * 0.012 * 1.05 + np.random.normal(0, 0.2, n),
                "final_price": demand * 4 + np.random.normal(0, 100, n),
            }
        )
        df["system_kw"] = np.maximum(df["system_kw"], 0.5)
        df["inverter_kw"] = np.maximum(df["inverter_kw"], 2)
        df["final_price"] = np.maximum(df["final_price"], 5000)

        models, metrics = self.model.train_models(df, cv_folds=3)
        self.model.models = models

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_save_models_creates_files(self):
        """Saving models creates checkpoint files."""
        self.model.save_models(self.temp_dir)

        # Check files exist
        self.assertTrue((Path(self.temp_dir) / "metadata.json").exists())
        for target in TARGETS:
            self.assertTrue((Path(self.temp_dir) / f"{target}_model.pkl").exists())
            self.assertTrue((Path(self.temp_dir) / f"{target}_importance.npy").exists())

    def test_load_models_restores_state(self):
        """Loading models restores training state."""
        self.model.save_models(self.temp_dir)

        # Create new model and load
        model_loaded = SolarQuoteModelFixed()
        model_loaded.load_models(self.temp_dir)

        # Check state restored
        self.assertEqual(model_loaded.panel_wattage, 700)
        self.assertEqual(len(model_loaded.models), len(TARGETS))
        self.assertIsNotNone(model_loaded.trained_at)
        self.assertEqual(len(model_loaded.metrics), len(TARGETS))

    def test_load_nonexistent_checkpoint(self):
        """Loading from nonexistent directory raises error."""
        model_new = SolarQuoteModelFixed()
        with self.assertRaises(Exception):
            model_new.load_models("/nonexistent/checkpoint")


class TestDesignSystemGeneration(unittest.TestCase):
    """Test main design_system() inference method."""

    def setUp(self):
        """Train a model for inference testing."""
        np.random.seed(42)
        n = 80
        self.model = SolarQuoteModelFixed()
        self.temp_dir = tempfile.mkdtemp()

        demand = np.random.uniform(200, 2000, n)
        df = pd.DataFrame(
            {
                "monthly_kwh_demand": demand,
                "peak_demand_kw": demand / 30 / 4,
                "region": np.random.choice(["north", "south", "east"], n),
                "roof_area_m2": np.random.uniform(20, 150, n),
                "battery_requested": np.random.randint(0, 2, n),
                "battery_kwh": np.random.uniform(0, 20, n),
                "system_kw": demand * 0.012 + np.random.normal(0, 0.3, n),
                "panel_wattage": 620,
                "inverter_kw": demand * 0.012 * 1.05 + np.random.normal(0, 0.2, n),
                "final_price": demand * 4 + np.random.normal(0, 100, n),
            }
        )
        df["system_kw"] = np.maximum(df["system_kw"], 0.5)
        df["inverter_kw"] = np.maximum(df["inverter_kw"], 2)
        df["final_price"] = np.maximum(df["final_price"], 5000)

        models, _ = self.model.train_models(df, cv_folds=3)
        self.model.models = models

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_design_system_basic(self):
        """Basic design generation produces required outputs."""
        result = self.model.design_system(monthly_kwh_demand=900)

        # Check output structure
        self.assertIn("system_kw", result)
        self.assertIn("inverter_kw", result)
        self.assertIn("panel_count", result)
        self.assertIn("estimated_price_central", result)
        self.assertIn("estimated_annual_production_kwh", result)

    def test_design_system_all_parameters(self):
        """Design with all parameters specified."""
        result = self.model.design_system(
            monthly_kwh_demand=1000,
            peak_demand_kw=10,
            region="south",
            roof_area_m2=80,
            battery_requested=1,
            battery_kwh=15,
        )

        self.assertIsInstance(result["system_kw"], (int, float))
        self.assertGreater(result["system_kw"], 0)
        self.assertIsInstance(result["inverter_kw"], (int, float))
        self.assertGreater(result["inverter_kw"], 0)

    def test_design_system_prices_positive(self):
        """Design generates non-negative price estimates."""
        result = self.model.design_system(monthly_kwh_demand=800)

        price_low, price_high = result["estimated_price_range"]
        self.assertGreaterEqual(price_low, 0)
        self.assertGreaterEqual(price_high, 0)
        self.assertLessEqual(price_low, price_high)

    def test_design_system_invalid_demand(self):
        """Design with invalid demand raises ValueError."""
        with self.assertRaises(ValueError):
            self.model.design_system(monthly_kwh_demand=np.nan)

        with self.assertRaises(ValueError):
            self.model.design_system(monthly_kwh_demand=-100)

    def test_design_system_no_models_raises_error(self):
        """Design without trained models raises RuntimeError."""
        empty_model = SolarQuoteModelFixed()
        with self.assertRaises(RuntimeError):
            empty_model.design_system(monthly_kwh_demand=800)

    def test_design_system_roof_constraint_applied(self):
        """Design respects roof area constraint."""
        # Specify small roof that can't fit large system
        result = self.model.design_system(
            monthly_kwh_demand=2000,
            roof_area_m2=10,  # Very small roof
        )

        is_valid = result["roof_area_validation"]["is_valid"]
        if not is_valid:
            # System size should be reduced
            self.assertLess(result["system_kw"], 5)

    def test_design_system_battery_affects_inverter(self):
        """Battery system increases inverter capacity."""
        result_no_battery = self.model.design_system(
            monthly_kwh_demand=900,
            battery_requested=0,
        )

        result_with_battery = self.model.design_system(
            monthly_kwh_demand=900,
            battery_requested=1,
            battery_kwh=15,
        )

        # Battery system should have >= inverter size
        self.assertGreaterEqual(
            result_with_battery["inverter_kw"],
            result_no_battery["inverter_kw"]
        )

    def test_design_system_confidence_bands(self):
        """Design produces confidence bands for system size."""
        result = self.model.design_system(monthly_kwh_demand=900)

        low, high = result["system_kw_confidence_range"]
        self.assertLess(low, high)
        self.assertLess(low, result["system_kw"])
        self.assertGreater(high, result["system_kw"])

    def test_design_system_panel_count_valid(self):
        """Panel count is realistic."""
        result = self.model.design_system(monthly_kwh_demand=900)

        panel_count = result["panel_count"]
        self.assertGreater(panel_count, 0)
        
        # Recalculated system_kw should match
        expected_kw = (panel_count * result["panel_wattage"]) / 1000
        self.assertAlmostEqual(result["system_kw"], expected_kw, places=1)

    def test_design_system_annual_production_reasonable(self):
        """Annual production estimate is reasonable."""
        result = self.model.design_system(
            monthly_kwh_demand=900,
            region="south",
        )

        annual_kwh = result["estimated_annual_production_kwh"]
        system_kw = result["system_kw"]
        
        # Rough check: 1 kW should produce ~1200-2000 kWh/year
        min_expected = system_kw * 1000
        max_expected = system_kw * 2500
        self.assertGreater(annual_kwh, min_expected)
        self.assertLess(annual_kwh, max_expected)

    def test_design_system_regional_irradiance_used(self):
        """Different regions have different irradiance values."""
        result_south = self.model.design_system(
            monthly_kwh_demand=900,
            region="south",
        )
        result_north = self.model.design_system(
            monthly_kwh_demand=900,
            region="north",
        )

        irradiance_south = result_south["regional_irradiance_kwh_m2_day"]
        irradiance_north = result_north["regional_irradiance_kwh_m2_day"]
        
        # South should have higher irradiance
        self.assertGreater(irradiance_south, irradiance_north)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Minimal trained model for edge case testing."""
        np.random.seed(42)
        n = 50
        self.model = SolarQuoteModelFixed()

        demand = np.full(n, 500.0)  # Constant demand
        df = pd.DataFrame(
            {
                "monthly_kwh_demand": demand,
                "peak_demand_kw": demand / 30 / 4,
                "region": ["south"] * n,
                "roof_area_m2": np.full(n, 100.0),
                "battery_requested": 0,
                "battery_kwh": 0,
                "system_kw": np.full(n, 6.0),
                "panel_wattage": 620,
                "inverter_kw": np.full(n, 6.5),
                "final_price": np.full(n, 20000),
            }
        )

        models, _ = self.model.train_models(df, cv_folds=3)
        self.model.models = models

    def test_zero_demand_handled(self):
        """Zero demand handled without crashing."""
        # Should not raise error
        result = self.model.design_system(monthly_kwh_demand=0.1)
        self.assertGreaterEqual(result["system_kw"], 0.5)  # Floor value

    def test_very_large_demand(self):
        """Very large demand handled."""
        result = self.model.design_system(monthly_kwh_demand=100000)
        self.assertGreater(result["system_kw"], 0)
        self.assertIsInstance(result["system_kw"], (int, float))

    def test_zero_roof_area(self):
        """Zero roof area handled."""
        result = self.model.design_system(
            monthly_kwh_demand=500,
            roof_area_m2=0,
        )
        # Should fail validation
        self.assertFalse(result["roof_area_validation"]["is_valid"])

    def test_unknown_region_default(self):
        """Unknown region defaults to standard irradiance."""
        result = self.model.design_system(
            monthly_kwh_demand=900,
            region="unknown_region",
        )
        # Should use default irradiance
        self.assertIn(result["regional_irradiance_kwh_m2_day"], REGIONAL_IRRADIANCE.values())


class TestPredictionFallback(unittest.TestCase):
    """Test fallback mechanisms when quantile models unavailable."""

    def setUp(self):
        """Train model with limited quantile training."""
        np.random.seed(42)
        n = 50
        self.model = SolarQuoteModelFixed()

        demand = np.random.uniform(200, 2000, n)
        df = pd.DataFrame(
            {
                "monthly_kwh_demand": demand,
                "peak_demand_kw": demand / 30 / 4,
                "region": np.random.choice(["north", "south"], n),
                "roof_area_m2": np.random.uniform(20, 150, n),
                "battery_requested": np.random.randint(0, 2, n),
                "battery_kwh": np.random.uniform(0, 20, n),
                "system_kw": demand * 0.012 + np.random.normal(0, 0.3, n),
                "panel_wattage": 620,
                "inverter_kw": demand * 0.012 * 1.05 + np.random.normal(0, 0.2, n),
                "final_price": demand * 4 + np.random.normal(0, 100, n),
            }
        )
        df["system_kw"] = np.maximum(df["system_kw"], 0.5)
        df["inverter_kw"] = np.maximum(df["inverter_kw"], 2)
        df["final_price"] = np.maximum(df["final_price"], 5000)

        models, _ = self.model.train_models(df, cv_folds=3, quantiles=[])
        self.model.models = models

    def test_fallback_when_quantile_models_absent(self):
        """When quantile models absent, percentage-based fallback used."""
        result = self.model.design_system(monthly_kwh_demand=900)

        # Should still produce confidence ranges
        low, high = result["system_kw_confidence_range"]
        self.assertLess(low, high)
        self.assertGreater(high, result["system_kw"])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
