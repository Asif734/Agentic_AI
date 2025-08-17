import json
import random
from faker import Faker

fake = Faker()

subjects = ["Math", "CS", "Physics", "Chemistry", "English", "History"]
grades = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-"]

data = []

for i in range(1, 51):  # 50 entries
    student_id = f"student_{1000 + i}"
    name = fake.first_name()
    email = fake.email()
    # Random grades for subjects
    student_grades = {subj: random.choice(grades) for subj in subjects}
    # Random schedule
    schedule_days = random.sample(["Mon", "Tue", "Wed", "Thu", "Fri"], k=3)
    schedule_time = f"{random.randint(8, 16)}:00 - {random.randint(9, 18)}:00"
    schedule = f"{', '.join(schedule_days)} {schedule_time}"
    
    entry = {
        "student_id": student_id,
        "name": name,
        "grades": student_grades,
        "schedule": schedule,
        "student_email": email
    }
    data.append(entry)

# Save to JSON
with open(r"C:\Users\Asif\VSCODE\University Chatbot\data\private_student_data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Private dataset with 50 entries created: data/private_data.json")
