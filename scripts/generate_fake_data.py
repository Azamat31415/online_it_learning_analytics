import pandas as pd
import numpy as np

def generate_fake_dataset(num_students=100):
    np.random.seed(42)

    data = {
        "user_id": range(1, num_students + 1),
        "avg_quiz_score": np.random.randint(50, 100, num_students),
        "num_submissions": np.random.randint(1, 20, num_students),
        "time_spent_hours": np.random.randint(5, 100, num_students),
        "forum_posts": np.random.randint(0, 15, num_students),
        "completion_status": np.random.choice([0, 1], num_students, p=[0.4, 0.6])
    }

    return pd.DataFrame(data)
