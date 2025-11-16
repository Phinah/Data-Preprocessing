"""
Create all sample placeholder files for team members
This script will create actual image and audio files with content
"""
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from image_processing import ImageProcessor
    from audio_processing import AudioProcessor
    
    print("="*70)
    print("CREATING ALL SAMPLE FILES")
    print("="*70)
    
    team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']
    
    # Create image samples
    print("\n[1/2] Creating sample images...")
    print("-" * 70)
    img_processor = ImageProcessor(base_dir='Images')
    for member in team_members:
        print(f"Creating images for {member}...")
        img_processor.create_sample_images(member)
    
    # Create audio samples
    print("\n[2/2] Creating sample audio files...")
    print("-" * 70)
    audio_processor = AudioProcessor(base_dir='Audio_data/raw', sr=16000)
    for member in team_members:
        print(f"Creating audio for {member}...")
        audio_processor.create_sample_audio(member, duration=2.0)
    
    print("\n" + "="*70)
    print("✓ ALL SAMPLE FILES CREATED SUCCESSFULLY!")
    print("="*70)
    print("\nCreated files:")
    print("\nImages:")
    for member in team_members:
        for expr in ['neutral', 'smile', 'surprised']:
            img_path = Path('Images') / f"{member}_{expr}.jpg"
            if img_path.exists():
                size = img_path.stat().st_size
                print(f"  ✓ {img_path.name} ({size} bytes)")
    
    print("\nAudio:")
    for member in team_members:
        for phrase in ['yes', 'confirm']:
            audio_path = Path('Audio_data/raw') / f"{member}_{phrase}.wav"
            if audio_path.exists():
                size = audio_path.stat().st_size
                print(f"  ✓ {audio_path.name} ({size} bytes)")
    print("="*70)
    
except ImportError as e:
    print("="*70)
    print("ERROR: Required modules not installed")
    print("="*70)
    print(f"\nMissing module: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    print("\nThen run this script again, or run:")
    print("  python image_processing.py  (creates image samples)")
    print("  python audio_processing.py   (creates audio samples)")
    print("="*70)
    sys.exit(1)

