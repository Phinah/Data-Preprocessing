"""
Audio Processing Pipeline for Voice Verification
Handles loading, augmentation, visualization, and feature extraction
"""
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """
    Audio processing pipeline for voice verification
    Handles loading, augmentation, visualization, and feature extraction
    """
    
    def __init__(self, base_dir='Audio_data/raw', sr=16000):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr  # Sample rate
        
    def load_audio(self, audio_path):
        """Load audio file"""
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        return y, sr
    
    def display_waveform(self, y, sr, title='Waveform'):
        """Display audio waveform"""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        return plt.gcf()
    
    def display_spectrogram(self, y, sr, title='Spectrogram'):
        """Display audio spectrogram"""
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_audio(self, audio_path, save_prefix='audio_viz'):
        """Create and save waveform and spectrogram visualizations"""
        y, sr = self.load_audio(audio_path)
        
        # Create specto_wave directory for visualizations (matching sample structure)
        viz_dir = Path('specto_wave')
        viz_dir.mkdir(exist_ok=True)
        
        # Waveform
        fig1 = self.display_waveform(y, sr, f'Waveform: {audio_path.name}')
        waveform_path = viz_dir / f'{save_prefix}_waveform.png'
        fig1.savefig(str(waveform_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Spectrogram
        fig2 = self.display_spectrogram(y, sr, f'Spectrogram: {audio_path.name}')
        spectrogram_path = viz_dir / f'{save_prefix}_spectrogram.png'
        fig2.savefig(str(spectrogram_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {waveform_path} and {spectrogram_path}")
    
    def augment_pitch_shift(self, y, sr, n_steps=2):
        """Shift pitch by n_steps semitones"""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    def augment_time_stretch(self, y, rate=1.2):
        """Stretch or compress time by rate factor"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def augment_add_noise(self, y, noise_level=0.005):
        """Add white noise to audio"""
        noise = np.random.randn(len(y))
        augmented = y + noise_level * noise
        return augmented
    
    def augment_change_speed(self, y, sr, speed_factor=1.1):
        """Change playback speed"""
        return librosa.effects.time_stretch(y, rate=speed_factor)
    
    def augment_add_reverb(self, y, sr, room_size=0.5):
        """Add simple reverb effect"""
        # Simple reverb using convolution with exponential decay
        reverb_time = int(sr * room_size)
        reverb = np.exp(-np.arange(reverb_time) / (sr * 0.1))
        reverb = reverb / np.sum(reverb)
        
        augmented = signal.convolve(y, reverb, mode='same')
        return augmented
    
    def extract_mfcc(self, y, sr, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Return mean and std of each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_spectral_features(self, y, sr):
        """Extract various spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_energy_features(self, y):
        """Extract energy-related features"""
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Total energy
        features['total_energy'] = np.sum(y ** 2)
        
        # Energy entropy
        frame_energies = librosa.feature.rms(y=y)[0]
        frame_energies = frame_energies / (np.sum(frame_energies) + 1e-10)
        features['energy_entropy'] = -np.sum(frame_energies * np.log2(frame_energies + 1e-10))
        
        return features
    
    def extract_chroma_features(self, y, sr):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])
    
    def extract_tempo(self, y, sr):
        """Extract tempo (BPM)"""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            return tempo[0] if hasattr(tempo, '__len__') else tempo
        except:
            return 120.0  # Default tempo
    
    def process_member_audio(self, member_name, phrases=['yes', 'confirm']):
        """
        Process all audio files for a team member
        Returns list of feature dictionaries
        """
        features_list = []
        
        for phrase in phrases:
            # Try different naming conventions
            audio_path = None
            for pattern in [
                f"{member_name}_{phrase}.wav",
                f"{member_name}-{phrase}.wav",
                f"{member_name}_{phrase}_approve.wav",
                f"{member_name}_{phrase}_transaction.wav"
            ]:
                test_path = self.base_dir / pattern
                if test_path.exists():
                    audio_path = test_path
                    break
            
            if audio_path is None or not audio_path.exists():
                print(f"Warning: {member_name} {phrase} audio not found, skipping...")
                continue
            
            # Load original audio
            y, sr = self.load_audio(audio_path)
            
            # Visualize original
            self.visualize_audio(audio_path, save_prefix=f'{member_name}_{phrase}')
            
            # Apply augmentations
            pitch_shifted = self.augment_pitch_shift(y, sr, n_steps=2)
            time_stretched = self.augment_time_stretch(y, rate=1.2)
            noisy = self.augment_add_noise(y, noise_level=0.005)
            speed_changed = self.augment_change_speed(y, sr, speed_factor=1.1)
            reverb = self.augment_add_reverb(y, sr, room_size=0.3)
            
            # Save augmented audio
            aug_dir = Path('Audio_data/augmented') / member_name
            aug_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(aug_dir / f"{phrase}_pitchup.wav"), pitch_shifted, sr)
            sf.write(str(aug_dir / f"{phrase}_fast.wav"), time_stretched, sr)
            sf.write(str(aug_dir / f"{phrase}_noise.wav"), noisy, sr)
            
            # Process all versions (original + augmented)
            versions = [
                ('original', y),
                ('pitchup', pitch_shifted),
                ('fast', time_stretched),
                ('noise', noisy),
            ]
            
            for aug_type, audio in versions:
                # Extract features
                mfcc_features = self.extract_mfcc(audio, sr)
                spectral_features = self.extract_spectral_features(audio, sr)
                energy_features = self.extract_energy_features(audio)
                chroma_features = self.extract_chroma_features(audio, sr)
                tempo = self.extract_tempo(audio, sr)
                
                # Create feature dictionary
                feature_dict = {
                    'member_name': member_name,
                    'phrase': phrase,
                    'augmentation': aug_type,
                    'audio_path': str(audio_path),
                    'duration': len(audio) / sr,
                    'tempo': tempo
                }
                
                # Add MFCC features
                for i, val in enumerate(mfcc_features):
                    feature_dict[f'mfcc_{i}'] = val
                
                # Add spectral features
                feature_dict.update(spectral_features)
                
                # Add energy features
                feature_dict.update(energy_features)
                
                # Add chroma features
                for i, val in enumerate(chroma_features):
                    feature_dict[f'chroma_{i}'] = val
                
                features_list.append(feature_dict)
            
            print(f"Processed {member_name} - {phrase}")
        
        return features_list
    
    def extract_features_from_file(self, audio_path, normalize=True):
        """
        Extract full feature set from a single WAV file.
        Used for real-time or batch verification.
        """
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = self.load_audio(audio_path)
        
        # Optional: normalize audio
        if normalize:
            y = librosa.util.normalize(y)

        # Extract all features
        mfcc = self.extract_mfcc(y, sr, n_mfcc=13)
        spectral = self.extract_spectral_features(y, sr)
        energy = self.extract_energy_features(y)
        chroma = self.extract_chroma_features(y, sr)
        tempo = self.extract_tempo(y, sr)

        # Build feature dictionary
        features = {
            'duration': len(y) / sr,
            'tempo': tempo,
        }

        # MFCC: 13 mean + 13 std = 26 values
        for i, val in enumerate(mfcc):
            features[f'mfcc_{i}'] = val

        # Spectral, energy
        features.update(spectral)
        features.update(energy)

        # Chroma: 12 mean + 12 std = 24 values
        for i, val in enumerate(chroma):
            features[f'chroma_{i}'] = val

        return features
    
    def create_sample_audio(self, member_name='sample_member', duration=2.0):
        """
        Create sample audio files for testing
        (Replace this with actual recordings)
        """
        phrases = ['yes', 'confirm']
        
        for phrase in phrases:
            # Generate a simple tone (replace with actual voice recording)
            t = np.linspace(0, duration, int(self.sr * duration))
            # Create a combination of frequencies to simulate voice
            frequency1 = 200 + np.random.randint(-50, 50)  # Base frequency
            frequency2 = 400 + np.random.randint(-50, 50)  # Harmonic
            
            audio = 0.3 * np.sin(2 * np.pi * frequency1 * t)
            audio += 0.2 * np.sin(2 * np.pi * frequency2 * t)
            
            # Add some noise to make it more realistic
            audio += 0.05 * np.random.randn(len(t))
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save
            filename = self.base_dir / f"{member_name}_{phrase}.wav"
            sf.write(str(filename), audio, self.sr)
            print(f"Created sample audio: {filename}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("AUDIO PROCESSING PIPELINE")
    print("="*60)
    
    processor = AudioProcessor(base_dir='Audio_data/raw', sr=16000)
    
    # Define your team members
    team_members = ['Phinah', 'Sage', 'Ayomide', 'Carine']
    
    # Check if audio files exist or are empty, create samples if needed
    print("\nChecking for audio files...")
    members_needing_samples = set()
    for member in team_members:
        for phrase in ['yes', 'confirm']:
            audio_path = processor.base_dir / f"{member}_{phrase}.wav"
            # Check if file doesn't exist or is empty (0 bytes)
            if not audio_path.exists() or (audio_path.exists() and audio_path.stat().st_size == 0):
                members_needing_samples.add(member)
                # Remove empty file if it exists
                if audio_path.exists() and audio_path.stat().st_size == 0:
                    audio_path.unlink()
    
    if members_needing_samples:
        print(f"\n⚠ Warning: Audio files missing or empty for: {', '.join(members_needing_samples)}")
        print("Creating sample placeholder audio files...")
        for member in members_needing_samples:
            processor.create_sample_audio(member, duration=2.0)
        print("Sample audio files created. Replace with actual recordings when available.\n")
    
    # Process all audio and extract features
    all_features = []
    
    for member in team_members:
        print(f"\nProcessing audio for {member}...")
        member_features = processor.process_member_audio(member)
        all_features.extend(member_features)
    
    # Create DataFrame and save to CSV
    if all_features:
        df_features = pd.DataFrame(all_features)
        df_features.to_csv('audio_features.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"Audio feature extraction complete!")
        print(f"Total samples processed: {len(all_features)}")
        print(f"Features saved to: audio_features.csv")
        print(f"Shape: {df_features.shape}")
        print(f"\nFeature columns:")
        print(df_features.columns.tolist()[:20], "...")
        print("="*60)
    else:
        print("No audio files were processed. Check your audio directory and file names.")

