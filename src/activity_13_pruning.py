# File: src/team_collaboration.py
"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) — Session 13
"""
import datetime
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

SOLAR_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "pruning_team1_solar.csv"
TARGET_COLUMN = "panel_fault_status"
CCP_ALPHA_VALUES = (0.001, 0.015, 0.080)


def split_solar_data():
    data = pd.read_csv(SOLAR_DATA_PATH)
    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def train_pruning_candidates(X_train, y_train):
    models = {}
    for ccp_alpha in CCP_ALPHA_VALUES:
        model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
        model.fit(X_train, y_train)
        models[ccp_alpha] = model
    return models

TEAM_REGISTRY = {
    "cohort": "Team 1",
    "repository": "data-mining-202660-team1",
    "members": [
        {
            "name": "Lourdes Auais Bulnes",
            "student_id": "00473471",
            "role": "Team Member",
            "assigned_reviewer": "Michel Jakob Häring",
            "git_feature_branch": "feature/activity-10-lourdes-auais",
            "preferred_ai_assistant": "Claude",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Teammates will append their dictionary blocks via their respective branches!
    ]
}

def display_team_roster():
    print(f"\n{'='*20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'='*20}")
    for m in TEAM_REGISTRY["members"]:
        print(f"* {m['name']} ({m['student_id']}) | Role: {m['role']} | Branch: {m['git_feature_branch']}")
    print('='*60 + '\n')

if __name__ == '__main__':
    display_team_roster()
    X_train, X_test, y_train, y_test = split_solar_data()
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    models = train_pruning_candidates(X_train, y_train)

    best_total_cost = None
    best_test_error = None

    for ccp_alpha, model in models.items():
        train_error = 1.0 - model.score(X_train, y_train)
        test_error = 1.0 - model.score(X_test, y_test)
        total_cost = test_error + ccp_alpha * model.get_n_leaves()

        if best_total_cost is None or total_cost < best_total_cost[1]:
            best_total_cost = (ccp_alpha, total_cost, test_error)
        if best_test_error is None or test_error < best_test_error[1]:
            best_test_error = (ccp_alpha, test_error, total_cost)

        print(
            f"ccp_alpha={ccp_alpha:.3f} | leaves={model.get_n_leaves()} "
            f"| depth={model.get_depth()} "
            f"| train_error={train_error:.3f} | test_error={test_error:.3f} "
            f"| total_cost={total_cost:.6f}"
        )

    print("\nBest by total cost:")
    print(
        f"ccp_alpha={best_total_cost[0]:.3f} | test_error={best_total_cost[2]:.3f} "
        f"| total_cost={best_total_cost[1]:.6f}"
    )
    print("\nLowest test error:")
    print(
        f"ccp_alpha={best_test_error[0]:.3f} | test_error={best_test_error[1]:.3f} "
        f"| total_cost={best_test_error[2]:.6f}"
    )