"""
Voice Verification Script
Used for system simulation and testing
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from audio_processing import AudioProcessor

# Load model and dependencies
model_dir = Path("models")
model_path = model_dir / "voice_verification_model.pkl"
encoder_path = model_dir / "voice_label_encoder.pkl"
feature_cols_path = model_dir / "voice_feature_columns.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Voice verification model not found: {model_path}\n"
                           "Please run train_audio_model.py first")

model = joblib.load(model_path)
feature_cols = joblib.load(feature_cols_path)

# Load encoder if it exists
if encoder_path.exists():
    le = joblib.load(encoder_path)
    has_encoder = True
else:
    has_encoder = False

processor = AudioProcessor(base_dir='Audio_data/raw', sr=16000)

print("="*60)
print("VOICE VERIFICATION")
print("="*60)
if has_encoder:
    print(f"Recognized speakers: {list(le.classes_)}")
print("="*60)

def verify_voice_file(audio_path, threshold=0.6):
    """
    Verify if an audio file matches any of the registered speakers.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the audio file to verify
    threshold : float
        Minimum probability threshold for acceptance
    
    Returns:
    --------
    recognized_speaker : str or None
        Name of recognized speaker, or None if not recognized
    confidence : float
        Confidence score (probability) of the recognition
    is_authorized : bool
        True if recognized with confidence above threshold
    """
    try:
        # Extract features
        feature_dict = processor.extract_features_from_file(str(audio_path))
        df = pd.DataFrame([feature_dict])
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[feature_cols].values.astype(np.float32)
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X)
        
        # Predict
        proba = model.predict_proba(X)[0]
        idx = proba.argmax()
        confidence = proba[idx]
        
        if has_encoder:
            recognized_speaker = le.classes_[idx]
        else:
            # Binary classification (authorized/unauthorized)
            recognized_speaker = "Authorized" if idx == 1 else "Unauthorized"
        
        # Check if above threshold
        is_authorized = confidence >= threshold
        
        return recognized_speaker, confidence, is_authorized
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, 0.0, False


# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python verify_voice.py <audio_path> [threshold]")
        print("\nExample:")
        print("  python verify_voice.py Audio_data/raw/Phinah_yes.wav")
        print("  python verify_voice.py Audio_data/raw/Carine_confirm.wav 0.7")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    
    print(f"\nVerifying audio: {audio_path}")
    print(f"Threshold: {threshold:.2f}\n")
    
    speaker, confidence, authorized = verify_voice_file(audio_path, threshold)
    
    print("="*60)
    if authorized:
        print(f"✓ ACCESS GRANTED")
        print(f"Recognized as: {speaker}")
        print(f"Confidence: {confidence:.1%}")
    else:
        if speaker:
            print(f"✗ ACCESS DENIED")
            print(f"Best match: {speaker} (confidence: {confidence:.1%})")
            print(f"Confidence below threshold ({threshold:.1%})")
        else:
            print(f"✗ ACCESS DENIED")
            print(f"Voice not recognized")
    print("="*60)

