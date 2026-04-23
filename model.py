from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None


CATEGORY_BANDS = {
    "Low": (0, 34),
    "Moderate": (35, 64),
    "High": (65, 100),
}
CLASS_SCORE_ANCHORS = {
    "Low": 20,
    "Moderate": 50,
    "High": 82,
}


def _normalize_label(label):
    return str(label).strip().lower().replace(" ", "_")


def infer_target_column(df):
    explicit_names = [
        "Stress_Level",
        "stress_level",
        "Stress_Score",
        "stress_score",
        "Stress",
        "stress",
    ]
    for name in explicit_names:
        if name in df.columns:
            return name

    stress_candidates = [column for column in df.columns if "stress" in column.lower()]
    if stress_candidates:
        return stress_candidates[-1]

    return df.columns[-1]


def infer_id_columns(df, target_column):
    id_columns = []
    for column in df.columns:
        if column == target_column:
            continue
        lowered = column.lower()
        if lowered.endswith("_id") or lowered == "id":
            id_columns.append(column)
    return id_columns


def infer_problem_type(target_series):
    if pd.api.types.is_numeric_dtype(target_series):
        unique_values = target_series.dropna().nunique()
        if unique_values <= 10:
            return "classification"
        return "regression"
    return "classification"


def build_preprocessor(features_df, scale_numeric=False):
    numeric_columns = features_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features_df.columns if column not in numeric_columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_candidate_models(problem_type, features_df):
    base_preprocessor = build_preprocessor(features_df, scale_numeric=False)
    scaled_preprocessor = build_preprocessor(features_df, scale_numeric=True)

    if problem_type == "classification":
        candidates = [
            (
                "RandomForestClassifier",
                Pipeline(
                    [
                        ("preprocessor", base_preprocessor),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=300,
                                random_state=42,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "GradientBoostingClassifier",
                Pipeline(
                    [
                        ("preprocessor", base_preprocessor),
                        ("model", GradientBoostingClassifier(random_state=42)),
                    ]
                ),
            ),
            (
                "LogisticRegression",
                Pipeline(
                    [
                        ("preprocessor", scaled_preprocessor),
                        ("model", LogisticRegression(max_iter=2000)),
                    ]
                ),
            ),
        ]
        if XGBClassifier is not None:
            candidates.append(
                (
                    "XGBoostClassifier",
                    Pipeline(
                        [
                            ("preprocessor", base_preprocessor),
                            (
                                "model",
                                XGBClassifier(
                                    n_estimators=250,
                                    max_depth=5,
                                    learning_rate=0.05,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    random_state=42,
                                    eval_metric="mlogloss",
                                ),
                            ),
                        ]
                    ),
                )
            )
        return candidates

    candidates = [
        (
            "RandomForestRegressor",
            Pipeline(
                [
                    ("preprocessor", base_preprocessor),
                    ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
                ]
            ),
        ),
        (
            "GradientBoostingRegressor",
            Pipeline(
                [
                    ("preprocessor", base_preprocessor),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
        ),
    ]
    if XGBRegressor is not None:
        candidates.append(
            (
                "XGBoostRegressor",
                Pipeline(
                    [
                        ("preprocessor", base_preprocessor),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=250,
                                max_depth=5,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                random_state=42,
                                objective="reg:squarederror",
                            ),
                        ),
                    ]
                ),
            )
        )
    return candidates


def _evaluate_model(problem_type, model, x_valid, y_valid):
    predictions = model.predict(x_valid)
    if problem_type == "classification":
        validation_score = accuracy_score(y_valid, predictions)
        return {
            "validation_score": float(validation_score),
            "accuracy": float(validation_score),
        }

    validation_score = r2_score(y_valid, predictions)
    rmse = mean_squared_error(y_valid, predictions, squared=False)
    return {
        "validation_score": float(validation_score),
        "r2": float(validation_score),
        "rmse": float(rmse),
    }


def train_model(dataset_path, model_path):
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    target_column = infer_target_column(df)
    id_columns = infer_id_columns(df, target_column)

    cleaned_df = df.copy()
    cleaned_df.columns = [str(column).strip() for column in cleaned_df.columns]
    target_column = infer_target_column(cleaned_df)
    id_columns = infer_id_columns(cleaned_df, target_column)

    cleaned_df = cleaned_df.dropna(axis=0, subset=[target_column]).copy()
    feature_df = cleaned_df.drop(columns=[target_column] + id_columns, errors="ignore")
    target_series = cleaned_df[target_column]

    problem_type = infer_problem_type(target_series)

    label_encoder = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(target_series):
        label_encoder = LabelEncoder()
        label_encoder.fit(target_series.unique())
        target_series = pd.Series(
            label_encoder.transform(target_series),
            index=target_series.index,
            name=target_series.name
        )

    stratify = target_series if problem_type == "classification" else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        feature_df,
        target_series,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    candidates = build_candidate_models(problem_type, feature_df)

    best_name = None
    best_model = None
    best_metrics = None
    best_score = -np.inf

    for model_name, pipeline in candidates:
        pipeline.fit(x_train, y_train)
        metrics = _evaluate_model(problem_type, pipeline, x_valid, y_valid)
        if metrics["validation_score"] > best_score:
            best_score = metrics["validation_score"]
            best_name = model_name
            best_model = pipeline
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("Unable to train a valid machine learning model.")

    artifact = {
        "dataset_path": str(dataset_path),
        "model_name": best_name,
        "model": best_model,
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_columns": feature_df.columns.tolist(),
        "id_columns": id_columns,
        "metrics": best_metrics,
        "target_min": float(target_series.min()) if pd.api.types.is_numeric_dtype(target_series) else None,
        "target_max": float(target_series.max()) if pd.api.types.is_numeric_dtype(target_series) else None,
        "label_encoder": label_encoder,
        "classes": [str(c) for c in (label_encoder.classes_ if label_encoder else sorted(target_series.astype(str).unique()))],
    }
    joblib.dump(artifact, model_path)
    return artifact


def load_or_train_model(dataset_path, model_path="saved_model.pkl"):
    model_path = Path(model_path)
    if model_path.exists():
        return joblib.load(model_path)
    return train_model(dataset_path, model_path)


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def project_ui_inputs_to_features(ui_inputs, feature_columns):
    projected = {}

    study = float(ui_inputs["study"])
    sleep = float(ui_inputs["sleep"])
    pressure = float(ui_inputs["pressure"])
    screen = float(ui_inputs["screen"])
    support = float(ui_inputs["support"])
    exercise = float(ui_inputs["exercise"])
    attendance = float(ui_inputs["attendance"])
    assign_load = float(ui_inputs["assign_load"])
    fin_pressure = float(ui_inputs["fin_pressure"])
    personal = float(ui_inputs["personal"])

    feature_values = {
        "Study_Hours_Per_Day": round(_clamp(study, 1, 12), 2),
        "Sleep_Hours_Per_Day": round(_clamp(sleep, 1, 12), 2),
        "Extracurricular_Hours_Per_Day": round(
            _clamp((screen * 0.35) + (exercise * 0.9) + (support * 0.25) + 0.8, 1, 12),
            2,
        ),
        "Social_Hours_Per_Day": round(
            _clamp((support * 0.75) + ((10 - personal) * 0.2) + ((attendance / 100) * 2.0), 1, 12),
            2,
        ),
        "Physical_Activity_Hours_Per_Day": round(
            _clamp((exercise * 1.15) + (max(0, 10 - pressure) * 0.2) + 0.8, 1, 12),
            2,
        ),
    }

    for column in feature_columns:
        if column in feature_values:
            projected[column] = feature_values[column]
        else:
            lowered = column.lower()
            if "study" in lowered:
                projected[column] = feature_values["Study_Hours_Per_Day"]
            elif "sleep" in lowered:
                projected[column] = feature_values["Sleep_Hours_Per_Day"]
            elif "social" in lowered:
                projected[column] = feature_values["Social_Hours_Per_Day"]
            elif "physical" in lowered or "activity" in lowered:
                projected[column] = feature_values["Physical_Activity_Hours_Per_Day"]
            elif "extra" in lowered:
                projected[column] = feature_values["Extracurricular_Hours_Per_Day"]
            elif "screen" in lowered:
                projected[column] = round(_clamp(screen, 0, 12), 2)
            elif "support" in lowered:
                projected[column] = round(_clamp(support, 1, 10), 2)
            elif "attendance" in lowered:
                projected[column] = round(_clamp(attendance, 0, 100), 2)
            elif "pressure" in lowered:
                projected[column] = round(_clamp(pressure, 1, 10), 2)
            elif "assignment" in lowered:
                projected[column] = round(_clamp(assign_load, 1, 10), 2)
            elif "financial" in lowered:
                projected[column] = round(_clamp(fin_pressure, 1, 10), 2)
            elif "personal" in lowered:
                projected[column] = round(_clamp(personal, 1, 10), 2)
            else:
                projected[column] = 0.0

    return pd.DataFrame([projected], columns=feature_columns)


def _probabilities_to_score(probabilities):
    expected_score = 0.0
    for label, probability in probabilities.items():
        normalized = _normalize_label(label)
        if normalized == "low":
            expected_score += probability * CLASS_SCORE_ANCHORS["Low"]
        elif normalized == "moderate":
            expected_score += probability * CLASS_SCORE_ANCHORS["Moderate"]
        elif normalized == "high":
            expected_score += probability * CLASS_SCORE_ANCHORS["High"]

    top_label = max(probabilities, key=probabilities.get)
    normalized_top = _normalize_label(top_label)
    if normalized_top == "low":
        band = CATEGORY_BANDS["Low"]
    elif normalized_top == "moderate":
        band = CATEGORY_BANDS["Moderate"]
    else:
        band = CATEGORY_BANDS["High"]

    return int(round(_clamp(expected_score, band[0], band[1])))


def _score_to_category(score):
    if score <= CATEGORY_BANDS["Low"][1]:
        return "Low"
    if score <= CATEGORY_BANDS["Moderate"][1]:
        return "Moderate"
    return "High"


def predict_from_ui_inputs(ui_inputs, artifact):
    feature_frame = project_ui_inputs_to_features(ui_inputs, artifact["feature_columns"])
    model = artifact["model"]
    problem_type = artifact["problem_type"]

    if problem_type == "classification":
        raw_predictions = model.predict(feature_frame)
        raw_probs = model.predict_proba(feature_frame)[0] if hasattr(model, "predict_proba") else None
        
        if artifact.get("label_encoder") is not None:
            predicted_label = artifact["label_encoder"].inverse_transform(raw_predictions)[0]
            class_labels = [str(c) for c in artifact["label_encoder"].classes_]
        else:
            predicted_label = str(raw_predictions[0])
            class_labels = artifact.get("classes", [str(label) for label in getattr(model, "classes_", [])])
        
        if raw_probs is not None:
            probabilities = {label: float(prob) for label, prob in zip(class_labels, raw_probs)}
        else:
            probabilities = {predicted_label: 1.0}

        score = _probabilities_to_score(probabilities)
        category = _score_to_category(score)
    else:
        raw_score = float(model.predict(feature_frame)[0])
        target_min = artifact.get("target_min", 0.0) or 0.0
        target_max = artifact.get("target_max", 100.0) or 100.0
        if target_max <= target_min:
            score = int(round(_clamp(raw_score, 0, 100)))
        else:
            scaled = ((raw_score - target_min) / (target_max - target_min)) * 100
            score = int(round(_clamp(scaled, 0, 100)))
        category = _score_to_category(score)
        predicted_label = category
        probabilities = {}

    return {
        "score": score,
        "category": category,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "metrics": artifact["metrics"],
        "model_name": artifact["model_name"],
        "problem_type": problem_type,
        "feature_frame": feature_frame,
    }
