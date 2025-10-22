import pandas as pd
import numpy as np

def generate_fake_dataset(num_students=100):
    np.random.seed(42)

    data = {
        "user_id": range(1, num_students + 1),
        "avg_quiz_score": np.random.randint(50, 100, num_students),
        "avg_total_grade": np.random.randint(40, 100, num_students),
        "num_submissions": np.random.randint(1, 20, num_students),
        "time_spent_hours": np.random.randint(5, 120, num_students),
        "attendance_rate": np.random.randint(50, 100, num_students),
        "num_resource_views": np.random.randint(10, 200, num_students),
        "late_submissions": np.random.randint(0, 5, num_students),
        "help_requests": np.random.randint(0, 10, num_students),
        "completion_status": np.random.choice([0, 1], num_students, p=[0.35, 0.65])
    }

    df = pd.DataFrame(data)
    return df
