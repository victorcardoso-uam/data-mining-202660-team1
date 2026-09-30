# File: src/team_collaboration.py
"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) — Session 13
"""
import datetime

TEAM_REGISTRY = {
    "cohort": "Team 1", # Update with your assigned team number
    "repository": "data-mining-202660-team1",
    "members": [
        {
            "name": "Your Full Name",
            "student_id": "00123456",
            "role": "Lead Data Engineer", # e.g., ML Engineer, Data Quality Auditor
            "assigned_reviewer": "Teammate Full Name",
            "git_feature_branch": "feature/activity-10-yourname",
            "preferred_ai_assistant": "GitHub Copilot in VS Code",
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
