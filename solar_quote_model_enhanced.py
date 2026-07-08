"""
Solar Power System Sizing Model - ENHANCED VERSION
=====================================================

Improved supervised model on historical quotation data to design viable
solar system estimates with uncertainty quantification, model persistence,
enhanced guardrails, and comprehensive evaluation.

--------------------------------------------------------------------------
EXPECTED HISTORICAL DATA SCHEMA
--------------------------------------------------------------------------
historical_quotes.csv should have one row per past quote, with columns:

    monthly_kwh_demand      : float  - customer's average monthly usage (kWh)
    peak_demand_kw          : float  - optional, peak load in kW (0 if unknown)
    region                  : str    - region/city code, used for price + irradiance
    roof_area_m2            : float  - available roof area in m^2 (or NaN if unknown)
    battery_requested       : int    - 1 if customer wanted battery backup, else 0
    battery_kwh             : float  - requested battery storage capacity (kWh)
    system_kw               : float  - TARGET: final quoted system capacity (kW)
    panel_wattage           : int    - wattage of panel model used in that quote
    inverter_kw             : float  - TARGET: final quoted inverter size (kW)
    final_price             : float  - TARGET: final quoted price

--------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import pickle
import json
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Feature definitions
NUMERIC_FEATURES = [
    "monthly_kwh_demand",
    "peak_demand_kw",
    "roof_area_m2",
    "battery_requested",
    "battery_kwh",
]
CATEGORICAL_FEATURES = ["region"]
TARGETS = ["system_kw", "inverter_kw", "final_price"]

# Standard panel wattages and inverter sizes available
AVAILABLE_PANEL_MODELS = {
    "standard_620w": 620,
    "high_eff_670w": 670,
    "high_eff_700w": 700,
}
DEFAULT_PANEL_WATTAGE = 620

AVAILABLE_INVERTER_SIZES_KW = [3, 5, 8, 10, 15, 20, 30, 50]

# Solar irradiance by region (kWh/m²/day) - to be refined with real data
REGIONAL_IRRADIANCE = {
    "north": 3.5,
    "south": 5.2,
    "east": 4.1,
    "west": 4.3,
    "default": 4.0,
}

# Battery system efficiency (DC to AC round-trip)
BATTERY_SYSTEM_EFFICIENCY = 0.85


class SolarQuoteModel:
    """
    Manages training, evaluation, and inference for solar system sizing.
    Supports model persistence and uncertainty quantification.
    """

    def __init__(self, panel_wattage: int = DEFAULT_PANEL_WATTAGE):
        """Initialize model container."""
        self.panel_wattage = panel_wattage
        self.models: Dict[str, Pipeline] = {}
        self.quantile_models: Dict[str, Pipeline] = {}  # For uncertainty quantification
        self.metrics: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, np.ndarray] = {}
        self.training_data_info: Dict = {}
        self.trained_at: Optional[str] = None

    def load_historical_data(
        self, path: str, drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Load historical quotation data from CSV.
        
        Args:
            path: Path to CSV file
            drop_na: Whether to drop rows with missing values
            
        Returns:
            DataFrame with expected columns
        """
        df = pd.read_csv(path)

        # Validate expected columns
        required_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Handle roof_area_m2 NaN values by imputing with median
        if "roof_area_m2" in df.columns and df["roof_area_m2"].isna().any():
            median_roof = df["roof_area_m2"].median()
            df["roof_area_m2"].fillna(median_roof, inplace=True)
            print(f"  [INFO] Imputed {df['roof_area_m2'].isna().sum()} missing roof_area_m2 values with median {median_roof:.1f} m²")

        # Handle battery_kwh NaN if present
        if "battery_kwh" in df.columns and df["battery_kwh"].isna().any():
            df["battery_kwh"].fillna(0, inplace=True)

        if drop_na:
            initial_rows = len(df)
            df = df.dropna()
            dropped = initial_rows - len(df)
            if dropped > 0:
                print(f"  [INFO] Dropped {dropped} rows with missing values")

        self.training_data_info = {
            "total_rows": len(df),
            "timestamp": datetime.now().isoformat(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "targets": TARGETS,
        }

        return df

    def build_pipeline(
        self, quantile: Optional[float] = None
    ) -> Pipeline:
        """
        Build feature preprocessing + regressor pipeline.
        
        Args:
            quantile: If provided, builds a quantile regressor for that quantile.
                     Otherwise builds standard GradientBoostingRegressor.
        
        Returns:
            Fitted sklearn Pipeline
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

        if quantile is not None:
            # Quantile regression for uncertainty bands
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            # Standard regression
            model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
            )

        return Pipeline(
            steps=[("preprocess", preprocessor), ("model", model)]
        )

    def train_models(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5,
        quantiles: list = None,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict]]:
        """
        Train regressors for each target with cross-validation and uncertainty quantification.
        
        Args:
            df: Training DataFrame
            test_size: Fraction for test split
            cv_folds: Number of cross-validation folds
            quantiles: List of quantiles for uncertainty bands (e.g., [0.1, 0.9])
            
        Returns:
            (fitted_models_dict, metrics_dict)
        """
        if quantiles is None:
            quantiles = [0.1, 0.9]

        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        fitted = {}
        self.metrics = {}

        print("\n" + "=" * 70)
        print("TRAINING MODELS WITH CROSS-VALIDATION")
        print("=" * 70)

        for target in TARGETS:
            print(f"\n--- Target: {target} ---")
            y = df[target].copy()

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Build and train main model
            pipe = self.build_pipeline()
            pipe.fit(X_train, y_train)

            # Predictions
            y_pred_train = pipe.predict(X_train)
            y_pred_test = pipe.predict(X_test)

            # Compute metrics
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

            # K-Fold Cross-Validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipe, X, y, cv=kf, scoring="neg_mean_absolute_error"
            )
            cv_mae_mean = -cv_scores.mean()
            cv_mae_std = cv_scores.std()

            self.metrics[target] = {
                "mae_train": mae_train,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "r2_test": r2_test,
                "mape_test": mape_test,
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
            }

            print(f"  Train MAE:        {mae_train:.4f}")
            print(f"  Test MAE:         {mae_test:.4f}")
            print(f"  Test RMSE:        {rmse_test:.4f}")
            print(f"  Test R²:          {r2_test:.4f}")
            print(f"  Test MAPE:        {mape_test:.4f}")
            print(f"  CV MAE (μ±σ):     {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")

            # Refit on full data for production
            pipe.fit(X, y)
            fitted[target] = pipe

            # Extract feature importance
            gb_model = pipe.named_steps["model"]
            self.feature_importance[target] = gb_model.feature_importances_

            # Train quantile models for uncertainty bands
            for q in quantiles:
                pipe_q = self.build_pipeline(quantile=q)
                pipe_q.fit(X, y)
                quantile_key = f"{target}_q{q}"
                self.quantile_models[quantile_key] = pipe_q

        self.trained_at = datetime.now().isoformat()
        return fitted, self.metrics

    def get_feature_importance(
        self, target: str, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Return top N most important features for a target.
        
        Args:
            target: Target name (e.g., 'system_kw')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if target not in self.feature_importance:
            raise ValueError(f"No feature importance for target: {target}")

        pipe = self.models[target]
        # Get feature names after transformation
        preprocess = pipe.named_steps["preprocess"]
        feature_names = (
            NUMERIC_FEATURES
            + list(
                preprocess.named_transformers_["cat"]
                .get_feature_names_out(CATEGORICAL_FEATURES)
            )
        )

        importances = self.feature_importance[target]
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return df_imp.head(top_n)

    def save_models(self, directory: str) -> None:
        """
        Save trained models, quantile models, and metadata to disk.
        
        Args:
            directory: Directory path to save models
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save main models
        for target, model in self.models.items():
            joblib.dump(model, path / f"{target}_model.pkl")

        # Save quantile models
        for key, model in self.quantile_models.items():
            joblib.dump(model, path / f"{key}_model.pkl")

        # Save metadata
        metadata = {
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "training_data_info": self.training_data_info,
            "panel_wattage": self.panel_wattage,
            "feature_importance_keys": list(self.feature_importance.keys()),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance arrays
        for target, importances in self.feature_importance.items():
            np.save(path / f"{target}_importance.npy", importances)

        print(f"✓ Models saved to {directory}")

    def load_models(self, directory: str) -> None:
        """
        Load trained models and metadata from disk.
        
        Args:
            directory: Directory path containing saved models
        """
        path = Path(directory)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.trained_at = metadata["trained_at"]
        self.metrics = metadata["metrics"]
        self.training_data_info = metadata["training_data_info"]
        self.panel_wattage = metadata["panel_wattage"]

        # Load main models
        for target in TARGETS:
            model_path = path / f"{target}_model.pkl"
            if model_path.exists():
                self.models[target] = joblib.load(model_path)

        # Load quantile models
        for model_file in path.glob("*_q*_model.pkl"):
            key = model_file.stem.replace("_model", "")
            self.quantile_models[key] = joblib.load(model_file)

        # Load feature importance
        for target in TARGETS:
            imp_path = path / f"{target}_importance.npy"
            if imp_path.exists():
                self.feature_importance[target] = np.load(imp_path)

        print(f"✓ Models loaded from {directory}")

    def snap_to_valid_inverter(self, predicted_kw: float) -> float:
        """Snap predicted inverter size to nearest real product."""
        return min(
            AVAILABLE_INVERTER_SIZES_KW,
            key=lambda x: abs(x - predicted_kw),
        )

    def validate_roof_area(
        self, roof_area_m2: float, system_kw: float
    ) -> Tuple[bool, str]:
        """
        Validate that roof area can fit the requested system.
        
        Args:
            roof_area_m2: Available roof area
            system_kw: Proposed system capacity
            
        Returns:
            (is_valid, message)
        """
        if roof_area_m2 is None or roof_area_m2 >= 999:
            return True, "Roof area not specified; design may need refinement"

        # Assume 180 W/m² (adjustable based on actual panel specs)
        watts_per_m2 = 180
        required_area = (system_kw * 1000) / watts_per_m2
        
        if required_area > roof_area_m2:
            return (
                False,
                f"System {system_kw} kW needs {required_area:.1f} m² but only {roof_area_m2} m² available",
            )
        
        return True, f"Roof area sufficient ({roof_area_m2} m² available, {required_area:.1f} m² required)"

    def calculate_battery_impact(
        self, battery_requested: int, battery_kwh: float, inverter_kw: float
    ) -> Tuple[float, str]:
        """
        Adjust inverter size based on battery system requirements.
        
        Battery systems require inverter oversizing for charge/discharge cycles.
        
        Args:
            battery_requested: 1 if battery wanted, 0 otherwise
            battery_kwh: Battery capacity in kWh
            inverter_kw: Base inverter size from model
            
        Returns:
            (adjusted_inverter_kw, note)
        """
        if battery_requested == 0:
            return inverter_kw, "No battery system"

        if battery_kwh <= 0:
            return inverter_kw, "Battery requested but capacity not specified"

        # Battery systems typically need inverter sized for 4-hour discharge at 1C rate
        # Plus overhead for efficiency and charge current
        discharge_current_kw = battery_kwh / 4.0  # 4-hour discharge
        battery_inverter_min = discharge_current_kw / BATTERY_SYSTEM_EFFICIENCY

        adjusted = max(inverter_kw, battery_inverter_min)
        note = f"Battery {battery_kwh} kWh: inverter bumped from {inverter_kw} to {adjusted:.1f} kW"

        return adjusted, note

    def design_system(
        self,
        monthly_kwh_demand: float,
        peak_demand_kw: float = 0,
        region: str = "default",
        roof_area_m2: float = 999,
        battery_requested: int = 0,
        battery_kwh: float = 0,
    ) -> Dict:
        """
        Main inference function: given customer demand + site info,
        returns a viable system design with uncertainty bands and guardrails.
        
        Args:
            monthly_kwh_demand: Average monthly energy demand (kWh)
            peak_demand_kw: Peak instantaneous demand (kW)
            region: Geographic region code
            roof_area_m2: Available roof area in m²
            battery_requested: 1 if battery backup wanted
            battery_kwh: Battery storage capacity (kWh)
            
        Returns:
            Dictionary with system design and confidence metrics
        """
        if not self.models:
            raise RuntimeError("Models not trained or loaded. Call train_models() first.")

        # Prepare input
        input_row = pd.DataFrame(
            [
                {
                    "monthly_kwh_demand": monthly_kwh_demand,
                    "peak_demand_kw": peak_demand_kw or (monthly_kwh_demand / 30 / 4),
                    "roof_area_m2": roof_area_m2,
                    "battery_requested": battery_requested,
                    "battery_kwh": battery_kwh,
                    "region": region,
                }
            ]
        )

        # Get point predictions
        raw_system_kw = float(self.models["system_kw"].predict(input_row)[0])
        raw_inverter_kw = float(
            self.models["inverter_kw"].predict(input_row)[0]
        )
        raw_price = float(self.models["final_price"].predict(input_row)[0])

        # Get uncertainty bands using quantile models
        system_kw_low = float(
            self.quantile_models["system_kw_q0.1"].predict(input_row)[0]
        )
        system_kw_high = float(
            self.quantile_models["system_kw_q0.9"].predict(input_row)[0]
        )

        price_low = float(
            self.quantile_models["final_price_q0.1"].predict(input_row)[0]
        )
        price_high = float(
            self.quantile_models["final_price_q0.9"].predict(input_row)[0]
        )

        # --- Apply Guardrails ---

        # 1. Roof area constraint
        max_kw_from_roof = (
            (roof_area_m2 * 0.18) if roof_area_m2 < 999 else raw_system_kw
        )
        system_kw = min(raw_system_kw, max_kw_from_roof)
        system_kw = max(system_kw, 0.5)  # Floor to avoid tiny systems

        # 2. Panel count and recomputation
        panel_count = max(
            1, round((system_kw * 1000) / self.panel_wattage)
        )
        system_kw = round(
            (panel_count * self.panel_wattage) / 1000, 2
        )  # Recompute from real panels

        # 3. Inverter sizing
        inverter_kw = self.snap_to_valid_inverter(raw_inverter_kw)

        # 4. Battery impact on inverter
        inverter_kw, battery_note = self.calculate_battery_impact(
            battery_requested, battery_kwh, inverter_kw
        )
        inverter_kw = self.snap_to_valid_inverter(inverter_kw)

        # 5. Validate roof area
        roof_valid, roof_note = self.validate_roof_area(roof_area_m2, system_kw)

        # 6. Price uncertainty band
        price_low = round(price_low, 2)
        price_high = round(price_high, 2)
        price_central = round(raw_price, 2)

        # Calculate confidence (narrower band = higher confidence)
        price_uncertainty_pct = (
            (price_high - price_low) / price_central * 100
            if price_central > 0
            else 0
        )

        # Solar irradiance factor for this region
        irradiance = REGIONAL_IRRADIANCE.get(region, REGIONAL_IRRADIANCE["default"])
        
        # Estimated annual production (kWh/year)
        estimated_annual_kwh = system_kw * irradiance * 365

        return {
            "system_kw": system_kw,
            "system_kw_confidence_range": (
                round(system_kw_low, 2),
                round(system_kw_high, 2),
            ),
            "panel_count": panel_count,
            "panel_wattage": self.panel_wattage,
            "inverter_kw": inverter_kw,
            "estimated_price_central": price_central,
            "estimated_price_range": (price_low, price_high),
            "price_uncertainty_pct": round(price_uncertainty_pct, 1),
            "estimated_annual_production_kwh": round(estimated_annual_kwh, 0),
            "regional_irradiance_kwh_m2_day": irradiance,
            "roof_area_validation": {
                "is_valid": roof_valid,
                "message": roof_note,
            },
            "battery_notes": battery_note if battery_requested else "Not requested",
            "design_notes": [
                f"Region: {region}",
                f"Monthly demand: {monthly_kwh_demand} kWh",
                f"Battery backup: {'Yes' if battery_requested else 'No'}",
            ],
        }

    def print_evaluation_report(self) -> None:
        """Print comprehensive evaluation report of all models."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)
        print(f"Trained at: {self.trained_at}")
        print(f"Panel wattage: {self.panel_wattage} W")
        print(f"Training data: {self.training_data_info['total_rows']} quotes\n")

        for target in TARGETS:
            if target not in self.metrics:
                continue

            m = self.metrics[target]
            print(f"📊 {target.upper()}")
            print(f"  Training MAE:          {m['mae_train']:.4f}")
            print(f"  Test MAE:              {m['mae_test']:.4f}")
            print(f"  Test RMSE:             {m['rmse_test']:.4f}")
            print(f"  Test R² Score:         {m['r2_test']:.4f}")
            print(f"  Test MAPE:             {m['mape_test']:.4f}")
            print(f"  Cross-val MAE (μ±σ):   {m['cv_mae_mean']:.4f} ± {m['cv_mae_std']:.4f}")
            print(f"  Samples (train/test):  {m['n_train_samples']}/{m['n_test_samples']}\n")


def main():
    """
    Demo: train models on synthetic data, save, load, and generate a design.
    """
    print("\n" + "=" * 70)
    print("SOLAR QUOTE MODEL - ENHANCED DEMO")
    print("=" * 70)

    # --- Generate synthetic training data ---
    print("\n[1] Generating synthetic historical data (300 quotes)...")
    rng = np.random.default_rng(42)
    n = 300

    demand = rng.uniform(200, 2000, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    roof = rng.uniform(20, 150, n)
    battery = rng.integers(0, 2, n)
    battery_kwh = battery * rng.uniform(5, 20, n)
    peak = demand / 30 / 4

    # Synthetic targets with realistic relationships
    system_kw = demand * 0.012 + rng.normal(0, 0.3, n)
    inverter_kw = system_kw * 1.05 + rng.normal(0, 0.2, n)
    price = system_kw * 900 + battery * 3000 + battery_kwh * 150 + rng.normal(0, 300, n)

    demo_df = pd.DataFrame(
        {
            "monthly_kwh_demand": demand,
            "peak_demand_kw": peak,
            "region": region,
            "roof_area_m2": roof,
            "battery_requested": battery,
            "battery_kwh": battery_kwh,
            "system_kw": np.maximum(system_kw, 0.5),  # Floor at 0.5 kW
            "inverter_kw": np.maximum(inverter_kw, 2),  # Floor at 2 kW
            "final_price": np.maximum(price, 5000),  # Floor at $5k
        }
    )

    # --- Initialize model and train ---
    print("\n[2] Initializing and training models...")
    model = SolarQuoteModel(panel_wattage=620)
    models, metrics = model.train_models(demo_df, cv_folds=5)
    model.models = models

    # Print evaluation report
    model.print_evaluation_report()

    # Print feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    for target in TARGETS:
        print(f"\n{target}:")
        imp_df = model.get_feature_importance(target, top_n=5)
        for _, row in imp_df.iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")

    # --- Save models ---
    print("\n[3] Saving trained models...")
    model.save_models("./solar_models_checkpoint")

    # --- Load models ---
    print("\n[4] Loading models from checkpoint...")
    model_loaded = SolarQuoteModel()
    model_loaded.load_models("./solar_models_checkpoint")

    # --- Generate system design ---
    print("\n[5] Generating system design for example customer...")
    print(
        "\n  Input: 900 kWh/month, 60 m² roof, south region, wants 10 kWh battery"
    )

    design = model_loaded.design_system(
        monthly_kwh_demand=900,
        peak_demand_kw=8,
        region="south",
        roof_area_m2=60,
        battery_requested=1,
        battery_kwh=10,
    )

    print("\n" + "=" * 70)
    print("SYSTEM DESIGN OUTPUT")
    print("=" * 70)
    for key, value in design.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, (list, tuple)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
