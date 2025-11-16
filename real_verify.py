"""
Complete System Simulation Script
Simulates the full authentication flow:
1. Facial Recognition → 2. Voice Verification → 3. Product Recommendation
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
from verify_face import verify_face_image
from verify_voice import verify_voice_file

# Load product recommendation model
model_dir = Path("models")
product_model_path = model_dir / "product_recommendation_model.pkl"
product_encoder_path = model_dir / "product_label_encoder.pkl"
product_features_path = model_dir / "product_feature_columns.pkl"

def predict_product(customer_id=None):
    """
    Predict product category for a customer
    Uses merged customer data
    """
    if not product_model_path.exists():
        print("Warning: Product recommendation model not found. Skipping product prediction.")
        return None
    
    try:
        model = joblib.load(product_model_path)
        le = joblib.load(product_encoder_path)
        feature_cols = joblib.load(product_features_path)
        
        # Load merged data to get customer features
        merged_path = Path('merge-output/merged_data.csv')
        if not merged_path.exists():
            print("Warning: Merged data not found. Cannot predict product.")
            return None
        
        df = pd.read_csv(merged_path)
        
        # If customer_id provided, use that customer's data
        if customer_id and customer_id in df['customer_id_new'].values:
            sample_row = df[df['customer_id_new'] == customer_id].iloc[0]
        else:
            # Use first available customer
            sample_row = df.iloc[0]
        
        # Prepare features
        X = sample_row[feature_cols].to_frame().T
        
        # Fill missing values
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        # Predict
        pred_encoded = model.predict(X)
        pred_label = le.inverse_transform(pred_encoded)[0]
        
        # Get probabilities
        proba = model.predict_proba(X)[0]
        confidence = proba.max()
        
        return pred_label, confidence
        
    except Exception as e:
        print(f"Error in product prediction: {e}")
        return None


def simulate_transaction(image_path, audio_path, face_threshold=0.6, voice_threshold=0.6):
    """
    Simulate a complete transaction flow
    
    Flow:
    1. Facial Recognition → 2. Voice Verification → 3. Product Recommendation
    """
    print("="*70)
    print("SYSTEM SIMULATION - COMPLETE TRANSACTION FLOW")
    print("="*70)
    
    # Step 1: Facial Recognition
    print("\n[STEP 1] FACIAL RECOGNITION")
    print("-" * 70)
    print(f"Processing image: {image_path}")
    
    face_member, face_confidence, face_authorized = verify_face_image(image_path, face_threshold)
    
    if not face_authorized:
        print("\n" + "="*70)
        print("✗ ACCESS DENIED - Facial Recognition Failed")
        print("="*70)
        if face_member:
            print(f"Best match: {face_member} (confidence: {face_confidence:.1%})")
            print(f"Confidence below threshold ({face_threshold:.1%})")
        else:
            print("Face not recognized")
        return False
    
    print(f"✓ Face recognized: {face_member} (confidence: {face_confidence:.1%})")
    print("→ Proceeding to voice verification...")
    
    # Step 2: Voice Verification
    print("\n[STEP 2] VOICE VERIFICATION")
    print("-" * 70)
    print(f"Processing audio: {audio_path}")
    
    voice_speaker, voice_confidence, voice_authorized = verify_voice_file(audio_path, voice_threshold)
    
    if not voice_authorized:
        print("\n" + "="*70)
        print("✗ ACCESS DENIED - Voice Verification Failed")
        print("="*70)
        if voice_speaker:
            print(f"Best match: {voice_speaker} (confidence: {voice_confidence:.1%})")
            print(f"Confidence below threshold ({voice_threshold:.1%})")
        else:
            print("Voice not recognized")
        return False
    
    print(f"✓ Voice verified: {voice_speaker} (confidence: {voice_confidence:.1%})")
    print("→ Proceeding to product recommendation...")
    
    # Step 3: Product Recommendation
    print("\n[STEP 3] PRODUCT RECOMMENDATION")
    print("-" * 70)
    
    product_result = predict_product()
    
    if product_result:
        product, product_confidence = product_result
        print(f"✓ Product predicted: {product} (confidence: {product_confidence:.1%})")
    else:
        print("⚠ Product prediction unavailable")
        product = "Unknown"
    
    # Final Result
    print("\n" + "="*70)
    print("✓ TRANSACTION COMPLETE - ACCESS GRANTED")
    print("="*70)
    print(f"Recognized User: {face_member}")
    print(f"Voice Verified: {voice_speaker}")
    print(f"Recommended Product: {product}")
    print("="*70)
    
    return True


def simulate_unauthorized_attempt(image_path=None, audio_path=None):
    """
    Simulate an unauthorized access attempt
    """
    print("="*70)
    print("SYSTEM SIMULATION - UNAUTHORIZED ACCESS ATTEMPT")
    print("="*70)
    
    if image_path:
        print("\n[TESTING] Unauthorized Face Image")
        print("-" * 70)
        face_member, face_confidence, face_authorized = verify_face_image(image_path, threshold=0.6)
        
        if face_authorized:
            print(f"⚠ WARNING: Unauthorized face was accepted! ({face_member}, {face_confidence:.1%})")
        else:
            print(f"✓ SECURITY: Unauthorized face correctly rejected")
            if face_member:
                print(f"   Best match: {face_member} (confidence: {face_confidence:.1%})")
    
    if audio_path:
        print("\n[TESTING] Unauthorized Voice Sample")
        print("-" * 70)
        voice_speaker, voice_confidence, voice_authorized = verify_voice_file(audio_path, threshold=0.6)
        
        if voice_authorized:
            print(f"⚠ WARNING: Unauthorized voice was accepted! ({voice_speaker}, {voice_confidence:.1%})")
        else:
            print(f"✓ SECURITY: Unauthorized voice correctly rejected")
            if voice_speaker:
                print(f"   Best match: {voice_speaker} (confidence: {voice_confidence:.1%})")
    
    print("\n" + "="*70)


# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python real_verify.py <image_path> <audio_path> [face_threshold] [voice_threshold]")
        print("\nExamples:")
        print("  # Full transaction simulation")
        print("  python real_verify.py Images/Phinah_neutral.jpg Audio_data/raw/Phinah_yes.wav")
        print("\n  # Unauthorized attempt simulation")
        print("  python real_verify.py --unauthorized Images/unknown.jpg Audio_data/raw/unknown.wav")
        sys.exit(1)
    
    if sys.argv[1] == '--unauthorized':
        # Unauthorized attempt
        image_path = sys.argv[2] if len(sys.argv) > 2 else None
        audio_path = sys.argv[3] if len(sys.argv) > 3 else None
        simulate_unauthorized_attempt(image_path, audio_path)
    else:
        # Full transaction
        image_path = sys.argv[1]
        audio_path = sys.argv[2]
        face_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
        voice_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
        
        simulate_transaction(image_path, audio_path, face_threshold, voice_threshold)

