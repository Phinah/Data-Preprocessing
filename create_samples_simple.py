"""
Simple script to create placeholder files - can be run after installing dependencies
Or manually create the files as needed
"""
from pathlib import Path

# Team members
team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']

# Create directories
Path('Images').mkdir(exist_ok=True)
Path('audio_raw').mkdir(exist_ok=True)

print("="*70)
print("SAMPLE FILE CREATION GUIDE")
print("="*70)
print("\nRequired files for each team member:")
print("\nIMAGES (in Images/ folder):")
for member in team_members:
    print(f"  {member}_neutral.jpg")
    print(f"  {member}_smile.jpg")
    print(f"  {member}_surprised.jpg")

print("\nAUDIO (in audio_raw/ folder):")
for member in team_members:
    print(f"  {member}_yes.wav")
    print(f"  {member}_confirm.wav")

print("\n" + "="*70)
print("To create sample placeholder files, run:")
print("  python3 create_sample_files.py")
print("\nOr install dependencies first:")
print("  pip install -r requirements.txt")
print("  python3 create_sample_files.py")
print("="*70)

