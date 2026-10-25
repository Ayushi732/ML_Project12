import os
import sys
import json
import pickle
import argparse
import math
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def validate_prediction_values(attendance, internal_marks, assignment_marks, study_hours):
    """Ensure prediction input values are numeric and within valid academic ranges."""
    values = {
        "Attendance": attendance,
        "Internal_Marks": internal_marks,
        "Assignment_Marks": assignment_marks,
        "Study_Hours": study_hours,
    }

    for name, value in values.items():
        if value is None or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a numeric value.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number.")

    limits = {
        "Attendance": (0, 100),
        "Internal_Marks": (0, 100),
        "Assignment_Marks": (0, 100),
        "Study_Hours": (0, 168),
    }

    for name, (minimum, maximum) in limits.items():
        value = values[name]
        if not minimum <= float(value) <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    mapping_path = os.path.join(models_dir, "cluster_mapping.json")
    
    if not (os.path.exists(kmeans_path) and os.path.exists(scaler_path) and os.path.exists(mapping_path)):
        raise FileNotFoundError("Model artifacts not found! Please run 'python src/train_model.py' first.")
        
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(mapping_path, "r") as f:
        cluster_mapping = json.load(f)
        
    return kmeans, scaler, cluster_mapping

def predict_student_cluster(attendance, internal_marks, assignment_marks, study_hours):
    validate_prediction_values(attendance, internal_marks, assignment_marks, study_hours)

    kmeans, scaler, cluster_mapping = load_artifacts()

    feature_cols = ["Attendance", "Internal_Marks", "Assignment_Marks", "Study_Hours"]
    features_df = pd.DataFrame([[attendance, internal_marks, assignment_marks, study_hours]], columns=feature_cols)
    scaled_features = scaler.transform(features_df)
    cluster_id = kmeans.predict(scaled_features)[0]

    cluster_info = cluster_mapping.get(str(cluster_id), {
        "name": f"Cluster {cluster_id}",
        "tier": "Unknown",
        "recommendation": "N/A"
    })

    return {
        "cluster_id": int(cluster_id),
        "group_name": cluster_info["name"],
        "tier": cluster_info["tier"],
        "recommendation": cluster_info["recommendation"],
        "input_data": {
            "Attendance": attendance,
            "Internal_Marks": internal_marks,
            "Assignment_Marks": assignment_marks,
            "Study_Hours": study_hours
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Predict Student Performance Cluster")
    parser.add_argument("--attendance", type=float, help="Attendance percentage (0-100)")
    parser.add_argument("--internal", type=float, help="Internal Exam Marks (0-100)")
    parser.add_argument("--assignment", type=float, help="Assignment Marks (0-100)")
    parser.add_argument("--hours", type=float, help="Study Hours per week (0-50)")
    
    args = parser.parse_args()
    
    # If parameters not provided in CLI, prompt interactively
    if None in [args.attendance, args.internal, args.assignment, args.hours]:
        print("=" * 60)
        print("--- STUDENT PERFORMANCE CLUSTERING INFERENCE ---")
        print("=" * 60)
        try:
            att = float(input("Enter Attendance Percentage (e.g. 78.5): "))
            intern = float(input("Enter Internal Marks out of 100 (e.g. 65.0): "))
            assign = float(input("Enter Assignment Marks out of 100 (e.g. 70.0): "))
            hrs = float(input("Enter Study Hours per week (e.g. 15.0): "))
        except (ValueError, KeyboardInterrupt):
            print("\n[!] Invalid input or cancelled. Exiting.")
            sys.exit(1)
    else:
        att = args.attendance
        intern = args.internal
        assign = args.assignment
        hrs = args.hours

    try:
        result = predict_student_cluster(att, intern, assign, hrs)
    except ValueError as error:
        print(f"\n[!] Invalid student data: {error}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("               PREDICTION RESULT")
    print("=" * 60)
    print(f"  Attendance:       {result['input_data']['Attendance']}%")
    print(f"  Internal Marks:   {result['input_data']['Internal_Marks']}/100")
    print(f"  Assignment Marks: {result['input_data']['Assignment_Marks']}/100")
    print(f"  Weekly Study:     {result['input_data']['Study_Hours']} hours/week")
    print("-" * 60)
    print(f"  Assigned Group:   {result['group_name'].upper()}")
    print(f"  Performance Tier: {result['tier']}")
    print(f"  Action Plan:      {result['recommendation']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
