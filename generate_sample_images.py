"""
Generate sample placeholder images for all team members
Creates actual image files with content (not empty)
"""
import cv2
import numpy as np
from pathlib import Path

def create_sample_image(member_name, expression, output_path):
    """Create a sample placeholder image with text"""
    # Create 400x400 colored image
    colors = {
        'neutral': (200, 200, 200),  # Gray
        'smile': (255, 200, 150),    # Light orange
        'surprised': (150, 200, 255)  # Light blue
    }
    
    color = colors.get(expression, (200, 200, 200))
    img = np.ones((400, 400, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    
    # Add text
    cv2.putText(img, member_name, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 
               1.5, (0, 0, 0), 3)
    cv2.putText(img, expression, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
               1.5, (0, 0, 0), 3)
    
    # Save as BGR (OpenCV format)
    cv2.imwrite(str(output_path), img)
    print(f"Created: {output_path.name}")

# Team members
team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']
expressions = ['neutral', 'smile', 'surprised']

# Create Images directory if it doesn't exist
images_dir = Path('Images')
images_dir.mkdir(exist_ok=True)

print("="*60)
print("GENERATING SAMPLE IMAGES")
print("="*60)
print(f"\nCreating images for: {', '.join(team_members)}")
print(f"Expressions: {', '.join(expressions)}\n")

# Generate all images
for member in team_members:
    for expr in expressions:
        filename = images_dir / f"{member}_{expr}.jpg"
        create_sample_image(member, expr, filename)

print("\n" + "="*60)
print("✓ All sample images created successfully!")
print("="*60)
print(f"\nCreated {len(team_members) * len(expressions)} image files:")
for member in team_members:
    for expr in expressions:
        img_path = images_dir / f"{member}_{expr}.jpg"
        if img_path.exists():
            size = img_path.stat().st_size
            print(f"  ✓ {img_path.name} ({size} bytes)")
print("="*60)

