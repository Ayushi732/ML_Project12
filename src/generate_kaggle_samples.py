import os
import numpy as np
import pandas as pd

def generate_kaggle_sample_datasets():
    np.random.seed(42)
    n_train = 500
    
    # 1. Generate Training Dataset (Realistic Student Performance)
    student_ids = [f"STU_{2024000 + i}" for i in range(1, n_train + 1)]
    gender = np.random.choice(["Male", "Female"], size=n_train, p=[0.48, 0.52])
    parental_education = np.random.choice(
        ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree"],
        size=n_train,
        p=[0.30, 0.35, 0.25, 0.10]
    )
    internet_access = np.random.choice(["Yes", "No"], size=n_train, p=[0.85, 0.15])
    extracurricular = np.random.choice(["Yes", "No"], size=n_train, p=[0.55, 0.45])
    tutoring_sessions = np.random.poisson(lam=2, size=n_train)
    
    study_hours = np.random.gamma(shape=4.0, scale=3.5, size=n_train).round(1) # avg ~14 hrs
    attendance = np.random.beta(a=8, b=2, size=n_train) * 60 + 40 # 40% - 100%
    attendance = attendance.round(1)
    
    previous_score = np.random.normal(loc=65, scale=14, size=n_train)
    previous_score = np.clip(previous_score, 25.0, 98.0).round(1)
    
    # Target calculations with realistic dependencies + noise
    education_boost = {"High School": 0, "Associate Degree": 3, "Bachelor's Degree": 6, "Master's Degree": 9}
    boost = np.array([education_boost[e] for e in parental_education])
    
    exam_score = (
        0.35 * previous_score +
        0.30 * attendance +
        0.75 * study_hours +
        1.5 * tutoring_sessions +
        boost +
        np.random.normal(loc=0, scale=4.5, size=n_train)
    )
    exam_score = np.clip(exam_score, 20.0, 100.0).round(1)
    
    # Categorical targets derived from exam score
    grades = []
    pass_fail = []
    for s in exam_score:
        if s >= 85:
            grades.append("A")
            pass_fail.append("Pass")
        elif s >= 70:
            grades.append("B")
            pass_fail.append("Pass")
        elif s >= 55:
            grades.append("C")
            pass_fail.append("Pass")
        elif s >= 45:
            grades.append("D")
            pass_fail.append("Fail")
        else:
            grades.append("F")
            pass_fail.append("Fail")
            
    df_train = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": gender,
        "Parental_Education": parental_education,
        "Internet_Access": internet_access,
        "Extracurricular": extracurricular,
        "Study_Hours_Weekly": study_hours,
        "Attendance_Rate": attendance,
        "Previous_Score": previous_score,
        "Tutoring_Sessions": tutoring_sessions,
        "Exam_Score": exam_score,
        "Final_Grade": grades,
        "Pass_Fail": pass_fail
    })
    
    # Introduce a few realistic NaN values (1-2%) to test data preprocessing & imputers
    mask_study = np.random.rand(n_train) < 0.02
    mask_att = np.random.rand(n_train) < 0.02
    mask_edu = np.random.rand(n_train) < 0.02
    df_train.loc[mask_study, "Study_Hours_Weekly"] = np.nan
    df_train.loc[mask_att, "Attendance_Rate"] = np.nan
    df_train.loc[mask_edu, "Parental_Education"] = np.nan
    
    # 2. Generate Batch Prediction Sample (Unlabelled new students, no targets)
    n_batch = 50
    batch_ids = [f"NEW_{3001 + i}" for i in range(n_batch)]
    b_gender = np.random.choice(["Male", "Female"], size=n_batch, p=[0.5, 0.5])
    b_parental_education = np.random.choice(
        ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree"],
        size=n_batch
    )
    b_internet_access = np.random.choice(["Yes", "No"], size=n_batch, p=[0.8, 0.2])
    b_extracurricular = np.random.choice(["Yes", "No"], size=n_batch, p=[0.5, 0.5])
    b_tutoring = np.random.poisson(lam=2, size=n_batch)
    b_study_hours = np.random.gamma(shape=4.0, scale=3.5, size=n_batch).round(1)
    b_attendance = (np.random.beta(a=8, b=2, size=n_batch) * 60 + 40).round(1)
    b_prev_score = np.clip(np.random.normal(loc=65, scale=14, size=n_batch), 30, 95).round(1)
    
    df_batch = pd.DataFrame({
        "Student_ID": batch_ids,
        "Gender": b_gender,
        "Parental_Education": b_parental_education,
        "Internet_Access": b_internet_access,
        "Extracurricular": b_extracurricular,
        "Study_Hours_Weekly": b_study_hours,
        "Attendance_Rate": b_attendance,
        "Previous_Score": b_prev_score,
        "Tutoring_Sessions": b_tutoring
    })
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "kaggle_student_performance_train.csv")
    batch_path = os.path.join(data_dir, "kaggle_student_prediction_batch.csv")
    
    df_train.to_csv(train_path, index=False)
    df_batch.to_csv(batch_path, index=False)
    
    print(f"Created training dataset: {train_path} ({len(df_train)} rows)")
    print(f"Created batch prediction sample: {batch_path} ({len(df_batch)} rows)")

if __name__ == "__main__":
    generate_kaggle_sample_datasets()
