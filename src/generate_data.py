import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples=600, random_state=42):
    """
    Generates a realistic student academic performance dataset with 3 underlying
    behavioral profiles (At-Risk, Average/Moderate, High Achievers) with natural variance.
    
    Features:
    - Student_ID: Unique identifier (e.g. STU1001)
    - Attendance_Rate: Attendance percentage (40% to 100%)
    - Study_Hours_Per_Week: Self-study hours per week (2 to 35 hours)
    - Internal_Marks: Midterm/internal exam marks (out of 100)
    - Assignment_Marks: Continuous assessment/assignment score (out of 100)
    """
    np.random.seed(random_state)
    
    n_at_risk = int(n_samples * 0.25)       # 25% ~ 150 students
    n_average = int(n_samples * 0.45)       # 45% ~ 270 students
    n_high = n_samples - n_at_risk - n_average # 30% ~ 180 students
    
    # 1. At-Risk / Needs Attention group
    att_risk = np.random.normal(loc=55, scale=7, size=n_at_risk)
    hours_risk = np.random.normal(loc=6, scale=2.5, size=n_at_risk)
    internal_risk = np.random.normal(loc=42, scale=7.5, size=n_at_risk)
    assign_risk = np.random.normal(loc=46, scale=8.0, size=n_at_risk)
    
    # 2. Average / Regular Performer group
    att_avg = np.random.normal(loc=76, scale=5.5, size=n_average)
    hours_avg = np.random.normal(loc=15, scale=3.5, size=n_average)
    internal_avg = np.random.normal(loc=68, scale=6.0, size=n_average)
    assign_avg = np.random.normal(loc=71, scale=6.5, size=n_average)
    
    # 3. High Achievers / Star Students group
    att_high = np.random.normal(loc=91, scale=4.0, size=n_high)
    hours_high = np.random.normal(loc=26, scale=4.0, size=n_high)
    internal_high = np.random.normal(loc=88, scale=4.5, size=n_high)
    assign_high = np.random.normal(loc=90, scale=4.5, size=n_high)
    
    # Concatenate groups
    attendance = np.concatenate([att_risk, att_avg, att_high])
    study_hours = np.concatenate([hours_risk, hours_avg, hours_high])
    internal_marks = np.concatenate([internal_risk, internal_avg, internal_high])
    assignment_marks = np.concatenate([assign_risk, assign_avg, assign_high])
    
    # Clip values to realistic academic boundaries
    attendance = np.clip(attendance, 35.0, 100.0).round(1)
    study_hours = np.clip(study_hours, 1.0, 40.0).round(1)
    internal_marks = np.clip(internal_marks, 20.0, 100.0).round(1)
    assignment_marks = np.clip(assignment_marks, 25.0, 100.0).round(1)
    
    # Shuffle order so groups are mixed naturally
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    student_ids = [f"STU{1001 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance": attendance[indices],
        "Internal_Marks": internal_marks[indices],
        "Assignment_Marks": assignment_marks[indices],
        "Study_Hours": study_hours[indices]
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "student_data.csv")
    
    df = generate_student_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created with {len(df)} records at: {csv_path}")
    print("\nSample Preview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe().round(2))
