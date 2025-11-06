import pandas as pd
import numpy as np

def generate_dataset(num_students=2000):
    np.random.seed(42)

    avg_quiz_score = np.random.randint(50, 100, num_students)
    time_spent_hours = np.random.randint(5, 120, num_students)
    attendance_rate = np.random.randint(50, 100, num_students)
    num_submissions = np.random.randint(1, 20, num_students)
    avg_total_grade = np.random.randint(40, 100, num_students)
    num_resource_views = np.random.randint(10, 200, num_students)
    late_submissions = np.random.randint(0, 5, num_students)
    help_requests = np.random.randint(0, 10, num_students)

    prob_success = (
            0.25 * (avg_quiz_score / 100) +
            0.25 * (attendance_rate / 100) +
            0.2 * (time_spent_hours / 120) +
            0.15 * (avg_total_grade / 100) +
            0.1 * (num_submissions / 20) -
            0.05 * (late_submissions / 5) +
            0.05 * (help_requests / 10)
    )

    prob_success = np.clip(prob_success, 0, 1)
    completion_status = [1 if np.random.rand() < p else 0 for p in prob_success]

    data = {
        "user_id": range(1, num_students + 1),
        "avg_quiz_score": avg_quiz_score,
        "avg_total_grade": avg_total_grade,
        "num_submissions": num_submissions,
        "time_spent_hours": time_spent_hours,
        "attendance_rate": attendance_rate,
        "num_resource_views": num_resource_views,
        "late_submissions": late_submissions,
        "help_requests": help_requests,
        "completion_status": completion_status
    }

    df = pd.DataFrame(data)
    return df
