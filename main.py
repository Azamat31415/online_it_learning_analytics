import pandas as pd
import joblib
from scripts.generate_data import generate_dataset
from scripts.predictor import train_predict_model
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    print("=== Online IT Learning Analytics ===")

    df = generate_dataset(num_students=2000)
    df.to_csv("data/fake_dataset.csv", index=False)
    print("Dataset saved to data/fake_dataset.csv")

    model = train_predict_model(df)
    model = joblib.load("models/trained_model.pkl")

    while True:
        user_input = input("\nEnter student ID (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        try:
            user_id = int(user_input)
            if user_id not in df["user_id"].values:
                print("User not found.")
                continue

            student_data = df[df["user_id"] == user_id]
            print("\nStudent data preview:")
            print(student_data.drop(columns=["user_id", "completion_status"]))

            features = student_data.drop(columns=["user_id", "completion_status"])
            prob = model.predict_proba(features)[0][1]
            print(f"Predicted success probability: {prob * 100:.1f}%")

        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()