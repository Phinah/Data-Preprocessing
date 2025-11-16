"""
Script to create sample placeholder files for all team members
Creates sample images and audio files with correct naming and extensions
"""
import cv2
import numpy as np
import soundfile as sf
from pathlib import Path
from image_processing import ImageProcessor
from audio_processing import AudioProcessor

def create_sample_images():
    """Create sample images for all team members"""
    print("="*60)
    print("CREATING SAMPLE IMAGES")
    print("="*60)
    
    processor = ImageProcessor(base_dir='Images')
    team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']
    
    for member in team_members:
        print(f"\nCreating sample images for {member}...")
        processor.create_sample_images(member)
    
    print("\n" + "="*60)
    print("Sample images created successfully!")
    print("="*60)
    print("\nCreated files:")
    for member in team_members:
        for expr in ['neutral', 'smile', 'surprised']:
            img_path = Path('Images') / f"{member}_{expr}.jpg"
            if img_path.exists():
                print(f"  ✓ {img_path}")


def create_sample_audio():
    """Create sample audio files for all team members"""
    print("\n" + "="*60)
    print("CREATING SAMPLE AUDIO FILES")
    print("="*60)
    
    processor = AudioProcessor(base_dir='audio_raw', sr=16000)
    team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']
    
    for member in team_members:
        print(f"\nCreating sample audio for {member}...")
        processor.create_sample_audio(member, duration=2.0)
    
    print("\n" + "="*60)
    print("Sample audio files created successfully!")
    print("="*60)
    print("\nCreated files:")
    for member in team_members:
        for phrase in ['yes', 'confirm']:
            audio_path = Path('audio_raw') / f"{member}_{phrase}.wav"
            if audio_path.exists():
                print(f"  ✓ {audio_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREATING SAMPLE PLACEHOLDER FILES")
    print("="*70)
    print("\nThis script creates placeholder files for:")
    print("  - Phinah")
    print("  - Sage")
    print("  - Ayomide")
    print("  - Carine")
    print("\nFiles will be created with correct naming and extensions.")
    print("="*70)
    
    # Create directories if they don't exist
    Path('Images').mkdir(exist_ok=True)
    Path('audio_raw').mkdir(exist_ok=True)
    
    # Create sample images
    create_sample_images()
    
    # Create sample audio
    create_sample_audio()
    
    print("\n" + "="*70)
    print("ALL SAMPLE FILES CREATED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Replace sample images with actual photos when available")
    print("2. Replace sample audio with actual recordings when available")
    print("3. Run: python image_processing.py")
    print("4. Run: python audio_processing.py")
    print("="*70)

