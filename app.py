import streamlit as st
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

import pulp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(page_title="AI Based Study Planner", layout="wide")
st.title("📚 AI-Driven Adaptive Study Planner")

# ===============================
# Data Models
# ===============================
@dataclass
class Topic:
    name: str
    difficulty: int
    estimated_hours: float
    priority: float

@dataclass
class Subject:
    name: str
    topics: List[Topic]

@dataclass
class StudentProfile:
    name: str
    daily_study_hours: float
    energy_level: str
    focus_score: float

@dataclass
class StudySession:
    subject: str
    topic: str
    duration: float
    date: datetime.date

@dataclass
class StudyPlan:
    sessions: List[StudySession] = field(default_factory=list)

@dataclass
class StudyFeedback:
    subject: str
    topic: str
    planned_hours: float
    actual_hours: float
    understanding: int

# ===============================
# Synthetic Data Generator
# ===============================
def generate_subjects(num_subjects=3, topics_per_subject=6):
    subjects = []
    for i in range(num_subjects):
        topics = []
        for j in range(topics_per_subject):
            difficulty = random.randint(1, 5)
            est_hours = round(np.random.uniform(1.5, 4.0) * difficulty, 2)
            priority = round(np.random.uniform(0.5, 1.5), 2)
            topics.append(Topic(f"Topic_{j+1}", difficulty, est_hours, priority))
        subjects.append(Subject(f"Subject_{i+1}", topics))
    return subjects

# ===============================
# Rule-Based Planner
# ===============================
def generate_rule_based_plan(subjects, student, exam_date):
    today = datetime.date.today()
    days = (exam_date - today).days
    all_topics = []

    for sub in subjects:
        for t in sub.topics:
            all_topics.append({
                "subject": sub.name,
                "topic": t.name,
                "hours": t.estimated_hours,
                "score": t.difficulty * t.priority
            })

    all_topics.sort(key=lambda x: x["score"], reverse=True)

    plan = StudyPlan()
    day = 0
    daily_left = student.daily_study_hours

    for t in all_topics:
        hrs = t["hours"]
        while hrs > 0 and day < days:
            used = min(hrs, daily_left)
            plan.sessions.append(
                StudySession(t["subject"], t["topic"], used, today + datetime.timedelta(days=day))
            )
            hrs -= used
            daily_left -= used
            if daily_left <= 0:
                day += 1
                daily_left = student.daily_study_hours

    return plan

# ===============================
# Optimization Planner (LP)
# ===============================
def optimize_study_plan(subjects, student, exam_date):
    today = datetime.date.today()
    days = [(today + datetime.timedelta(days=i)) for i in range((exam_date - today).days)]

    topics = []
    for s in subjects:
        for t in s.topics:
            topics.append({
                "id": f"{s.name}_{t.name}",
                "subject": s.name,
                "topic": t.name,
                "hours": t.estimated_hours,
                "weight": t.priority * t.difficulty
            })

    model = pulp.LpProblem("StudyPlanner", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x", ((t["id"], d) for t in topics for d in days), lowBound=0
    )

    model += pulp.lpSum(x[(t["id"], d)] * t["weight"] for t in topics for d in days)

    for d in days:
        model += pulp.lpSum(x[(t["id"], d)] for t in topics) <= student.daily_study_hours

    for t in topics:
        model += pulp.lpSum(x[(t["id"], d)] for d in days) == t["hours"]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    plan = StudyPlan()
    for t in topics:
        for d in days:
            val = x[(t["id"], d)].value()
            if val and val > 0:
                plan.sessions.append(
                    StudySession(t["subject"], t["topic"], round(val, 2), d)
                )
    return plan

# ===============================
# Feedback & ML
# ===============================
def simulate_feedback(plan):
    feedback = []
    for s in plan.sessions:
        actual = round(s.duration * np.random.uniform(0.7, 1.1), 2)
        feedback.append(
            StudyFeedback(s.subject, s.topic, s.duration, actual, random.randint(2, 5))
        )
    return feedback

def train_ml_model(feedback, subjects, student):
    diff, pr = {}, {}
    for s in subjects:
        for t in s.topics:
            diff[t.name] = t.difficulty
            pr[t.name] = t.priority

    data = []
    for f in feedback:
        data.append([
            diff[f.topic], pr[f.topic], f.planned_hours, student.focus_score, f.actual_hours
        ])

    df = pd.DataFrame(data, columns=["difficulty", "priority", "planned", "focus", "actual"])
    X = df.drop("actual", axis=1)
    y = df["actual"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    return model

# ===============================
# Visualization Helpers
# ===============================
def plan_df(plan):
    return pd.DataFrame([{
        "date": s.date,
        "subject": s.subject,
        "hours": s.duration
    } for s in plan.sessions])

# ===============================
# Streamlit UI
# ===============================
st.sidebar.header("Student Inputs")
daily_hours = st.sidebar.slider("Daily Study Hours", 2.0, 8.0, 4.0)
focus = st.sidebar.slider("Focus Score", 0.5, 1.0, 0.8)
exam_days = st.sidebar.slider("Days Until Exam", 15, 120, 45)

if st.sidebar.button("Generate Study Plan"):
    student = StudentProfile("User", daily_hours, "evening", focus)
    subjects = generate_subjects()
    exam_date = datetime.date.today() + datetime.timedelta(days=exam_days)

    rule_plan = generate_rule_based_plan(subjects, student, exam_date)
    opt_plan = optimize_study_plan(subjects, student, exam_date)

    feedback = simulate_feedback(opt_plan)
    ml_model = train_ml_model(feedback, subjects, student)

    st.subheader("📅 Plan Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Rule-Based Planner")
        st.dataframe(plan_df(rule_plan).head(15))

    with col2:
        st.markdown("### AI-Optimized Planner")
        st.dataframe(plan_df(opt_plan).head(15))

    st.subheader("📊 Daily Study Load")
    fig, ax = plt.subplots()
    plan_df(rule_plan).groupby("date")["hours"].sum().plot(ax=ax, label="Rule-Based")
    plan_df(opt_plan).groupby("date")["hours"].sum().plot(ax=ax, label="AI-Optimized")
    ax.legend()
    ax.set_ylabel("Hours")
    st.pyplot(fig)

