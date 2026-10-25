"""
Modular Machine Learning Engine for Dynamic Student Performance Prediction.
Supports dynamic dataset ingestion, automatic task detection (Regression vs Classification),
robust preprocessing pipelines, multi-model training/evaluation, permutation importance,
and batch/single inference.
"""

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Metrics & Explainability
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance


def detect_identifier_columns(df: pd.DataFrame) -> list:
    """
    Detect likely identifier/ID columns (e.g. Student_ID, Roll_No, Name, Email)
    to safely exclude them from default feature selection.
    """
    id_pattern = re.compile(
        r"(?i)(^id$|_id$|id_|student_id|roll|roll_no|rollno|name|email|uuid|index|serial|reg_no|registration)"
    )
    identified = []
    
    for col in df.columns:
        col_str = str(col).strip()
        if id_pattern.search(col_str):
            identified.append(col)
        elif df[col].dtype == object and df[col].nunique() == len(df) and len(df) > 10:
            identified.append(col)
            
    return identified


def detect_task_type(df: pd.DataFrame, target_col: str) -> dict:
    """
    Analyzes target column and automatically determines whether the task
    is Regression (numerical marks/scores) or Classification (grades, pass/fail, tiers).
    """
    if target_col not in df.columns:
        return {"task": "regression", "reason": "Target not found, defaulting to regression"}
        
    series = df[target_col].dropna()
    n_unique = series.nunique()
    dtype = series.dtype
    
    # If non-numeric (string, boolean, category, object) -> Classification
    if (
        pd.api.types.is_object_dtype(dtype) or
        pd.api.types.is_string_dtype(dtype) or
        pd.api.types.is_bool_dtype(dtype) or
        pd.api.types.is_categorical_dtype(dtype)
    ):
        return {
            "task": "classification",
            "n_unique": n_unique,
            "classes": sorted([str(x) for x in series.unique()[:20]]),
            "reason": f"Target contains text/categorical labels with {n_unique} distinct classes."
        }
        
    # If numeric
    if pd.api.types.is_numeric_dtype(dtype):
        # If integer with 7 or fewer unique values (e.g., binary 0/1 or grade bands 1-5)
        if n_unique <= 7 and np.all(np.equal(np.mod(series.values, 1), 0)):
            return {
                "task": "classification",
                "n_unique": n_unique,
                "classes": sorted(list(series.unique())),
                "reason": f"Target is discrete numeric with only {n_unique} unique integer values (e.g., encoded classes)."
            }
        else:
            return {
                "task": "regression",
                "n_unique": n_unique,
                "range": (float(series.min()), float(series.max())),
                "mean": float(series.mean()),
                "reason": f"Target is continuous numeric (Range: {series.min():.1f} to {series.max():.1f}, {n_unique} unique values)."
            }
            
    return {"task": "regression", "n_unique": n_unique, "reason": "Defaulting to regression based on numeric properties."}


def separate_feature_types(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Separates selected feature columns into numeric and categorical lists.
    """
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Builds a robust scikit-learn ColumnTransformer that handles missing values,
    scales numerical features, and one-hot encodes categorical features.
    """
    transformers = []
    
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_cols))
        
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))
        
    return ColumnTransformer(transformers=transformers, remainder="drop")


def find_semantic_columns(df: pd.DataFrame) -> dict:
    """
    Searches for academic semantic columns in any uploaded dataset
    (marks/scores, attendance, study hours, pass/fail, grades) to power dynamic dashboard KPIs.
    """
    semantic = {
        "marks_col": None,
        "attendance_col": None,
        "study_hours_col": None,
        "pass_fail_col": None,
        "grade_col": None
    }
    
    marks_pattern = re.compile(r"(?i)(score|marks|exam|grade|total|final|test|point|gpa|percentage)")
    att_pattern = re.compile(r"(?i)(attendance|attend|presence|regularity)")
    study_pattern = re.compile(r"(?i)(study|hour|hours|duration|time)")
    pass_pattern = re.compile(r"(?i)(pass|fail|result|status|outcome)")
    grade_pattern = re.compile(r"(?i)(grade|tier|band|category|level)")

    for col in df.columns:
        col_lower = str(col).lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            if att_pattern.search(col_lower) and not semantic["attendance_col"]:
                semantic["attendance_col"] = col
            elif study_pattern.search(col_lower) and not semantic["study_hours_col"]:
                semantic["study_hours_col"] = col
            elif marks_pattern.search(col_lower) and not semantic["marks_col"]:
                semantic["marks_col"] = col
        else:
            if pass_pattern.search(col_lower) and not semantic["pass_fail_col"]:
                semantic["pass_fail_col"] = col
            elif grade_pattern.search(col_lower) and not semantic["grade_col"]:
                semantic["grade_col"] = col

    # Fallback for marks: if no marks_col matched pattern, pick the first numeric column with variance
    if not semantic["marks_col"]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 5]
        if numeric_cols:
            semantic["marks_col"] = numeric_cols[-1] # Usually targets are near the end

    return semantic


def get_available_algorithms(task_type: str) -> list:
    """
    Returns list of supported algorithms for the given task type.
    """
    if task_type == "regression":
        return [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "K-Nearest Neighbors Regressor",
            "Support Vector Regressor (SVR)"
        ]
    else:
        return [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors Classifier",
            "Support Vector Classifier (SVC)",
            "Gaussian Naive Bayes"
        ]


def train_supervised_models(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    task_type: str,
    selected_models: list = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    End-to-end training and evaluation pipeline.
    Trains selected or all suitable models (Regression or Classification), computes metrics,
    and returns a trained state dictionary ready for inference.
    """
    # 1. Clean data: drop rows where target is missing
    clean_df = df.dropna(subset=[target_col]).copy()
    
    if len(clean_df) < 10:
        raise ValueError("The dataset contains fewer than 10 valid records after removing missing targets.")
        
    X = clean_df[feature_cols].copy()
    y = clean_df[target_col].copy()
    
    numeric_cols, categorical_cols = separate_feature_types(X, feature_cols)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No valid numerical or categorical features selected for training.")
        
    # 2. Target Encoding if Classification
    label_encoder = None
    target_classes = None
    
    if task_type == "classification":
        y_str = y.astype(str)
        if y_str.nunique() < 2:
            raise ValueError(
                f"Classification target '{target_col}' contains only 1 unique class ('{y_str.iloc[0]}'). "
                "Supervised classification requires at least 2 distinct classes."
            )
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_str)
        target_classes = list(label_encoder.classes_)
        y_final = y_encoded
    else:
        # Ensure numerical target
        y_final = pd.to_numeric(y, errors="coerce")
        valid_idx = ~y_final.isna()
        X = X.loc[valid_idx]
        y_final = y_final.loc[valid_idx].values
        if len(y_final) < 10:
            raise ValueError(f"Target '{target_col}' contains fewer than 10 valid numerical records.")
        
    # 3. Train/Test Split
    stratify_arg = None
    if task_type == "classification":
        class_counts = pd.Series(y_final).value_counts()
        if (class_counts >= 2).all() and len(class_counts) > 1:
            stratify_arg = y_final
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_final, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    
    # 4. Model Catalog
    k_neighbors = min(5, max(1, len(X_train) - 1))
    
    if task_type == "regression":
        model_catalog = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=random_state),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=k_neighbors),
            "Support Vector Regressor (SVR)": SVR(C=1.0, epsilon=0.2)
        }
    else:
        model_catalog = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=random_state),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=k_neighbors),
            "Support Vector Classifier (SVC)": SVC(probability=True, random_state=random_state),
            "Gaussian Naive Bayes": GaussianNB()
        }
        
    # Filter by user selection if provided
    if selected_models and len(selected_models) > 0:
        models_to_train = {k: v for k, v in model_catalog.items() if k in selected_models}
        if not models_to_train:
            models_to_train = model_catalog
    else:
        models_to_train = model_catalog

    trained_pipelines = {}
    metrics_records = []
    predictions_dict = {}
    
    # 5. Train each model with an independent ColumnTransformer pipeline
    for name, model_estimator in models_to_train.items():
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])
        
        # Fit on training data ONLY to prevent leakage
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        trained_pipelines[name] = pipeline
        predictions_dict[name] = y_pred
        
        if task_type == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_test, y_pred)
            
            metrics_records.append({
                "Model": name,
                "MAE (Lower Better)": round(mae, 3),
                "MSE (Lower Better)": round(mse, 3),
                "RMSE (Lower Better)": round(rmse, 3),
                "R2 Score (Higher Better)": round(r2, 4)
            })
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            metrics_records.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision (Weighted)": round(prec, 4),
                "Recall (Weighted)": round(rec, 4),
                "F1-Score (Higher Better)": round(f1, 4)
            })
            
    metrics_df = pd.DataFrame(metrics_records)
    
    # Determine best model based on primary evaluation metric
    if task_type == "regression":
        best_model_name = metrics_df.sort_values(by="R2 Score (Higher Better)", ascending=False).iloc[0]["Model"]
    else:
        best_model_name = metrics_df.sort_values(by="F1-Score (Higher Better)", ascending=False).iloc[0]["Model"]
        
    return {
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "pipelines": trained_pipelines,
        "best_model_name": best_model_name,
        "metrics_df": metrics_df,
        "predictions_dict": predictions_dict,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "target_classes": target_classes,
        "total_trained": len(clean_df),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


def predict_single_student(trained_state: dict, student_input: dict, model_name: str = None) -> dict:
    """
    Predicts marks, grade, or pass/fail outcome for an individual student record.
    """
    if model_name is None or model_name not in trained_state["pipelines"]:
        model_name = trained_state["best_model_name"]
        
    pipeline = trained_state["pipelines"][model_name]
    feature_cols = trained_state["feature_cols"]
    
    # Construct 1-row DataFrame matching the exact training column structure
    input_df = pd.DataFrame([student_input])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
            
    input_df = input_df[feature_cols]
    
    raw_pred = pipeline.predict(input_df)[0]
    probabilities = None
    
    if trained_state["task_type"] == "classification":
        model_step = pipeline.named_steps["model"]
        if hasattr(model_step, "predict_proba"):
            try:
                proba_array = pipeline.predict_proba(input_df)[0]
                classes = trained_state["target_classes"]
                probabilities = {str(classes[i]): round(float(proba_array[i]), 4) for i in range(len(classes))}
            except Exception:
                probabilities = None
                
        if trained_state["label_encoder"] is not None:
            decoded_pred = trained_state["label_encoder"].inverse_transform([int(raw_pred)])[0]
        else:
            decoded_pred = raw_pred
            
        return {
            "prediction": decoded_pred,
            "raw_prediction": raw_pred,
            "probabilities": probabilities,
            "model_used": model_name,
            "task_type": "classification",
            "target_col": trained_state["target_col"]
        }
    else:
        return {
            "prediction": round(float(raw_pred), 2),
            "raw_prediction": float(raw_pred),
            "model_used": model_name,
            "task_type": "regression",
            "target_col": trained_state["target_col"]
        }


def predict_batch_students(trained_state: dict, batch_df: pd.DataFrame, model_name: str = None) -> tuple:
    """
    Predicts target values for a bulk uploaded CSV containing multiple students.
    If the target column is present in the batch CSV, also calculates comparison metrics!
    """
    if model_name is None or model_name not in trained_state["pipelines"]:
        model_name = trained_state["best_model_name"]
        
    pipeline = trained_state["pipelines"][model_name]
    feature_cols = trained_state["feature_cols"]
    target_col = trained_state["target_col"]
    
    missing_cols = [c for c in feature_cols if c not in batch_df.columns]
    if missing_cols:
        return None, f"Uploaded batch file is missing required feature columns: {missing_cols}", None
        
    X_batch = batch_df[feature_cols].copy()
    raw_preds = pipeline.predict(X_batch)
    
    output_df = batch_df.copy()
    pred_col_name = f"Predicted_{target_col}"
    
    if trained_state["task_type"] == "classification" and trained_state["label_encoder"] is not None:
        decoded_preds = trained_state["label_encoder"].inverse_transform(raw_preds.astype(int))
        output_df[pred_col_name] = decoded_preds
    else:
        if trained_state["task_type"] == "regression":
            output_df[pred_col_name] = np.round(raw_preds, 2)
        else:
            output_df[pred_col_name] = raw_preds

    # Evaluate batch metrics if target column is already in the batch CSV with ground truth
    batch_metrics = None
    if target_col in batch_df.columns:
        valid_targets = batch_df[target_col].dropna()
        if len(valid_targets) > 0 and len(valid_targets) == len(batch_df):
            try:
                if trained_state["task_type"] == "regression":
                    y_true = pd.to_numeric(batch_df[target_col], errors="coerce").values
                    y_p = output_df[pred_col_name].values
                    valid_mask = ~np.isnan(y_true)
                    if valid_mask.sum() > 0:
                        batch_metrics = {
                            "MAE": round(mean_absolute_error(y_true[valid_mask], y_p[valid_mask]), 3),
                            "RMSE": round(float(np.sqrt(mean_squared_error(y_true[valid_mask], y_p[valid_mask]))), 3),
                            "R2": round(r2_score(y_true[valid_mask], y_p[valid_mask]), 4)
                        }
                else:
                    y_true = batch_df[target_col].astype(str).values
                    y_p = output_df[pred_col_name].astype(str).values
                    batch_metrics = {
                        "Accuracy": round(accuracy_score(y_true, y_p), 4),
                        "F1-Score": round(f1_score(y_true, y_p, average="weighted", zero_division=0), 4)
                    }
            except Exception:
                batch_metrics = None
            
    return output_df, None, batch_metrics


def get_feature_importances(trained_state: dict, model_name: str = None) -> pd.DataFrame:
    """
    Extracts feature importances or coefficients from the selected model.
    Falls back to permutation importance for algorithms without native coefficients
    (e.g., K-Nearest Neighbors, Support Vector Machines, Gaussian Naive Bayes).
    """
    if model_name is None or model_name not in trained_state["pipelines"]:
        model_name = trained_state["best_model_name"]
        
    pipeline = trained_state["pipelines"][model_name]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    try:
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [re.sub(r"^(num__|cat__)", "", name) for name in feature_names]
    except Exception:
        clean_names = trained_state["feature_cols"]
        
    importances = None
    
    # 1. Native feature importances (Tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # 2. Coefficients (Linear and Logistic models)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    # 3. Permutation Importance fallback (KNN, SVR, SVC, Naive Bayes)
    else:
        try:
            X_test = trained_state["X_test"]
            y_test = trained_state["y_test"]
            # Fast permutation importance on test set
            perm = permutation_importance(pipeline, X_test, y_test, n_repeats=3, random_state=42)
            # Aggregate importance back to original input features
            df_perm = pd.DataFrame({
                "Feature": trained_state["feature_cols"],
                "Importance": np.maximum(0, perm.importances_mean)
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            return df_perm
        except Exception:
            importances = None
            
    if importances is not None and len(importances) == len(clean_names):
        df_imp = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df_imp
        
    return pd.DataFrame()


def generate_academic_suggestions(trained_state: dict, student_input: dict, predicted_value) -> list:
    """
    Generates sensible, cautious academic suggestions based on observed feature values.
    Explicitly labeled as heuristic guidelines rather than guaranteed causal conclusions.
    """
    suggestions = []
    
    # Check for attendance
    for k, v in student_input.items():
        k_lower = str(k).lower()
        if "attend" in k_lower:
            try:
                val = float(v)
                if val < 75.0:
                    suggestions.append(f"📅 **Attendance Consideration:** Current recorded attendance is {val:.1f}%. Institutional data shows regular attendance (≥75%) strongly correlates with academic stability.")
                elif val >= 90.0:
                    suggestions.append(f"⭐ **Attendance Strength:** High attendance rate ({val:.1f}%) provides a strong learning foundation.")
            except (ValueError, TypeError):
                pass
                
        # Check for study hours
        if "study" in k_lower or "hour" in k_lower:
            try:
                val = float(v)
                if val < 10.0:
                    suggestions.append(f"📖 **Self-Study Rhythm:** Weekly study commitment is currently {val:.1f} hrs/week. Structuring consistent 12–15 hours weekly can help reinforce lecture concepts.")
                elif val >= 25.0:
                    suggestions.append(f"📚 **Dedicated Study Routine:** High weekly study volume ({val:.1f} hrs/week). Ensure balanced time management to avoid pre-exam fatigue.")
            except (ValueError, TypeError):
                pass
                
        # Check for previous marks
        if "prev" in k_lower or "internal" in k_lower:
            try:
                val = float(v)
                if val < 50.0:
                    suggestions.append(f"📝 **Prior Marks Focus:** Prior assessments indicate foundational topics may require remedial revision or targeted mentoring.")
            except (ValueError, TypeError):
                pass

    if not suggestions:
        suggestions.append("ℹ️ Maintain steady coursework pacing and seek periodic instructor feedback before midterm milestones.")

    return suggestions
