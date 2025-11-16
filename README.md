# User Identity and Product Recommendation System

A complete multimodal authentication and recommendation system that combines facial recognition, voice verification, and product recommendation using machine learning.

## Table of Contents

- [System Overview](#system-overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Usage Pipeline](#usage-pipeline)
- [System Flow](#system-flow)
- [Model Details](#model-details)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Team Members](#team-members)

## System Overview

This system implements a sequential authentication and recommendation flow:

1. **Facial Recognition** → Verifies user identity from facial images
2. **Voice Verification** → Confirms identity through voice samples  
3. **Product Recommendation** → Predicts product category based on customer data

The system uses machine learning models (Random Forest, Logistic Regression, XGBoost) to perform multi-modal authentication and personalized product recommendations.

## Features

### Image Processing
- **Augmentations**: Rotation, flipping, grayscale conversion, brightness adjustment, noise addition
- **Feature Extraction**: 
  - Histogram features
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Pattern)
  - Color moments
- **Automatic Processing**: Processes all team member images with multiple augmentations

### Audio Processing
- **Augmentations**: Pitch shift, time stretch, noise addition, speed change, reverb
- **Feature Extraction**:
  - MFCC (13 coefficients)
  - Spectral features (centroid, rolloff, bandwidth)
  - Energy features (RMS, total energy, entropy)
  - Chroma features
  - Tempo estimation
- **Visualization**: Automatic generation of waveforms and spectrograms
- **Automatic Processing**: Processes all team member audio with multiple augmentations

### Machine Learning Models
- **Facial Recognition**: Random Forest / Logistic Regression
- **Voice Verification**: Random Forest
- **Product Recommendation**: Random Forest / XGBoost
- **Evaluation Metrics**: Accuracy, F1-Score, Log Loss

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the repository**
   ```bash
   cd Data-Preprocessing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, librosa, sklearn; print('All dependencies installed successfully!')"
   ```

## Project Structure

```
Data-Preprocessing/
├── Images/                          # Facial images directory
│   ├── {member}_neutral.jpg         # Neutral expression images
│   ├── {member}_smile.jpg           # Smiling expression images
│   ├── {member}_surprised.jpg       # Surprised expression images
│   └── augmented/                   # Augmented images (auto-generated)
│       └── {member}/                # Per-member augmented images
│
├── Audio_data/                      # Audio data directory
│   ├── raw/                         # Original audio recordings
│   │   ├── {member}_yes.wav         # "Yes" phrase recordings
│   │   └── {member}_confirm.wav     # "Confirm" phrase recordings
│   └── augmented/                   # Augmented audio (auto-generated)
│       └── {member}/                 # Per-member augmented audio
│
├── models/                          # Trained models directory
│   ├── face_recognition_model.pkl
│   ├── face_label_encoder.pkl
│   ├── face_feature_columns.pkl
│   ├── voice_verification_model.pkl
│   ├── voice_label_encoder.pkl
│   ├── voice_feature_columns.pkl
│   ├── product_recommendation_model.pkl
│   ├── product_label_encoder.pkl
│   └── product_feature_columns.pkl
│
├── merge-output/                    # Merged dataset and EDA outputs
│   ├── merged_data.csv              # Merged customer data
│   ├── merge_validation.txt         # Validation report
│   └── plot_*.png                   # EDA visualizations
│
├── specto_wave/                     # Audio visualizations (auto-generated)
│   ├── {member}_{phrase}_waveform.png
│   └── {member}_{phrase}_spectrogram.png
│
├── scripts/                         # Processing scripts
│   ├── merge_datasets.py           # Dataset merging and feature engineering
│   ├── product_recommendation.py   # Product recommendation model training
│   └── predict.py                  # Standalone product prediction
│
├── image_processing.py              # Image feature extraction pipeline
├── audio_processing.py              # Audio feature extraction pipeline
├── train_face_model.py              # Train facial recognition model
├── train_audio_model.py             # Train voice verification model
├── verify_face.py                   # Face verification script
├── verify_voice.py                  # Voice verification script
├── real_verify.py                   # Complete system simulation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Data Preparation

### 1. Image Data
Each team member should have 3 facial images with the following naming convention:
- `{member_name}_neutral.jpg` - Neutral facial expression
- `{member_name}_smile.jpg` - Smiling facial expression
- `{member_name}_surprised.jpg` - Surprised facial expression

**Place images in:** `Images/` directory

**Supported formats:** JPG, JPEG

**Example:**
```
Images/
├── Phinah_neutral.jpg
├── Phinah_smile.jpg
├── Phinah_surprised.jpg
├── Sage_neutral.jpg
└── ...
```

### 2. Audio Data
Each team member should have 2 audio recordings with the following naming convention:
- `{member_name}_yes.wav` - Recording saying "Yes" or "Yes, approve"
- `{member_name}_confirm.wav` - Recording saying "Confirm" or "Confirm transaction"

**Place audio files in:** `Audio_data/raw/` directory

**Supported formats:** WAV (recommended), other formats will be converted

**Sample rate:** 16 kHz (automatically handled during processing)

**Example:**
```
Audio_data/raw/
├── Phinah_yes.wav
├── Phinah_confirm.wav
├── Sage_yes.wav
└── ...
```

### 3. Customer Datasets
Two CSV files are required for product recommendation:
- `customer_social_profiles - customer_social_profiles.csv` - Customer social media profiles
- `customer_transactions - customer_transactions.csv` - Customer transaction history

**Place CSV files in:** Project root directory

**Note:** The scripts will automatically handle ID mapping and merging.

## Usage Pipeline

### Step 1: Merge Customer Datasets
Merge customer social profiles and transactions, perform feature engineering, and generate EDA visualizations:

```bash
python scripts/merge_datasets.py
```

**Outputs:**
- `merge-output/merged_data.csv` - Merged and engineered dataset
- `merge-output/merge_validation.txt` - Validation report with statistics
- `merge-output/plot_purchase_amount_dist.png` - Purchase amount distribution
- `merge-output/plot_box_by_category.png` - Box plots by category
- `merge-output/plot_correlations.png` - Feature correlation matrix

### Step 2: Extract Image Features
Process all facial images, apply augmentations, and extract features:

```bash
python image_processing.py
```

**Outputs:**
- `image_features.csv` - Extracted image features for all members
- `Images/augmented/{member}/` - Augmented images (rotated, flipped, grayscale, bright, noisy)
- `sample_images_display.png` - Sample image visualization (if display is enabled)

**What it does:**
- Loads all team member images
- Applies 5 types of augmentations per image
- Extracts histogram, HOG, LBP, and color moment features
- Saves augmented images and feature CSV

### Step 3: Extract Audio Features
Process all audio recordings, apply augmentations, and extract features:

```bash
python audio_processing.py
```

**Outputs:**
- `audio_features.csv` - Extracted audio features for all members
- `Audio_data/augmented/{member}/` - Augmented audio files (pitchup, fast, noise)
- `specto_wave/{member}_{phrase}_waveform.png` - Waveform visualizations
- `specto_wave/{member}_{phrase}_spectrogram.png` - Spectrogram visualizations

**What it does:**
- Loads all team member audio files
- Applies 4 types of augmentations per audio
- Extracts MFCC, spectral, energy, chroma, and tempo features
- Generates waveform and spectrogram visualizations

### Step 4: Train Models

#### Train Facial Recognition Model
```bash
python train_face_model.py
```

**Outputs:**
- `models/face_recognition_model.pkl` - Trained model
- `models/face_label_encoder.pkl` - Label encoder
- `models/face_feature_columns.pkl` - Feature column names

**Evaluation:** Prints accuracy, F1-score, and classification report

#### Train Voice Verification Model
```bash
python train_audio_model.py
```

**Outputs:**
- `models/voice_verification_model.pkl` - Trained model
- `models/voice_label_encoder.pkl` - Label encoder
- `models/voice_feature_columns.pkl` - Feature column names

**Evaluation:** Prints accuracy, F1-score, and classification report

#### Train Product Recommendation Model
```bash
python scripts/product_recommendation.py
```

**Outputs:**
- `models/product_recommendation_model.pkl` - Trained model (Random Forest)
- `models/product_model_xgb.joblib` - XGBoost model (if XGBoost available)
- `models/product_label_encoder.pkl` - Label encoder
- `models/product_feature_columns.pkl` - Feature column names

**Evaluation:** Prints accuracy, F1-score, and log loss

### Step 5: System Simulation

#### Full Transaction Simulation
Simulate a complete authentication and recommendation flow:

```bash
python real_verify.py Images/Phinah_neutral.jpg Audio_data/raw/Phinah_yes.wav
```

**With custom thresholds:**
```bash
python real_verify.py Images/Phinah_neutral.jpg Audio_data/raw/Phinah_yes.wav 0.7 0.7
```

**What it does:**
1. Verifies facial recognition (threshold: 0.6 default)
2. Verifies voice (threshold: 0.6 default)
3. Predicts product recommendation
4. Displays complete transaction result

#### Unauthorized Access Attempt
Test security by simulating unauthorized access:

```bash
python real_verify.py --unauthorized Images/unknown.jpg Audio_data/raw/unknown.wav
```

**What it does:**
- Tests if unauthorized faces/voices are correctly rejected
- Displays security warnings if unauthorized access is accepted

#### Individual Verification

**Face Verification:**
```bash
python verify_face.py Images/Phinah_neutral.jpg
```

**Voice Verification:**
```bash
python verify_voice.py Audio_data/raw/Phinah_yes.wav
```

#### Product Prediction
Predict product category for a customer:

```bash
python scripts/predict.py
```

## System Flow

```
┌─────────────────────┐
│   User Image        │
│   (Input)           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Facial             │ ──✗ Fail → ACCESS DENIED
│  Recognition        │
│  (Step 1)           │
└──────────┬──────────┘
           │ ✓ Pass
           ▼
┌─────────────────────┐
│  Product            │
│  Recommendation     │
│  (Prepared)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Voice              │ ──✗ Fail → ACCESS DENIED
│  Verification       │
│  (Step 2)           │
└──────────┬──────────┘
           │ ✓ Pass
           ▼
┌─────────────────────┐
│  Display            │
│  Predicted          │
│  Product            │
│  (Success)          │
└─────────────────────┘
```

## Model Details

### Facial Recognition Model
- **Algorithm**: Random Forest Classifier (primary), Logistic Regression (alternative)
- **Features**: Histogram, HOG, LBP, Color moments
- **Input**: Image features extracted from facial images
- **Output**: Team member identity with confidence score
- **Evaluation**: Accuracy, F1-Score (weighted)

### Voice Verification Model
- **Algorithm**: Random Forest Classifier
- **Features**: MFCC, Spectral, Energy, Chroma, Tempo
- **Input**: Audio features extracted from voice recordings
- **Output**: Speaker identity with confidence score
- **Evaluation**: Accuracy, F1-Score (weighted)

### Product Recommendation Model
- **Algorithm**: Random Forest Classifier (primary), XGBoost (alternative)
- **Features**: Customer purchase history, social media engagement, ratings, sentiment
- **Input**: Merged customer data with engineered features
- **Output**: Product category recommendation with confidence score
- **Evaluation**: Accuracy, F1-Score (weighted), Log Loss

## Output Files

### Feature Files
- `image_features.csv` - Extracted image features (all members, all augmentations)
- `audio_features.csv` - Extracted audio features (all members, all augmentations)
- `merge-output/merged_data.csv` - Merged customer data with engineered features

### Model Files
- `models/face_recognition_model.pkl` - Facial recognition model
- `models/face_label_encoder.pkl` - Face label encoder
- `models/face_feature_columns.pkl` - Face feature column names
- `models/voice_verification_model.pkl` - Voice verification model
- `models/voice_label_encoder.pkl` - Voice label encoder
- `models/voice_feature_columns.pkl` - Voice feature column names
- `models/product_recommendation_model.pkl` - Product recommendation model
- `models/product_label_encoder.pkl` - Product label encoder
- `models/product_feature_columns.pkl` - Product feature column names

### Visualization Files
- `merge-output/plot_purchase_amount_dist.png` - Purchase amount distribution
- `merge-output/plot_box_by_category.png` - Box plots by category
- `merge-output/plot_correlations.png` - Feature correlation matrix
- `specto_wave/{member}_{phrase}_waveform.png` - Audio waveform visualizations
- `specto_wave/{member}_{phrase}_spectrogram.png` - Audio spectrogram visualizations
- `sample_images_display.png` - Sample image display (if generated)

### Augmented Data
- `Images/augmented/{member}/` - Augmented images (5 types per original image)
- `Audio_data/augmented/{member}/` - Augmented audio files (3 types per original audio)

## Troubleshooting

### Missing Images/Audio Files
**Problem:** Scripts report missing files or skip processing

**Solutions:**
- Ensure files follow naming convention: `{member}_{expression}.jpg` or `{member}_{phrase}.wav`
- Check file paths are correct (Images/ for images, Audio_data/raw/ for audio)
- Verify file extensions are correct (.jpg, .wav)
- The scripts will automatically create placeholder files if missing (for initial setup)

### Model Not Found Errors
**Problem:** `FileNotFoundError` when running verification scripts

**Solutions:**
- Run training scripts before verification:
  ```bash
  python train_face_model.py
  python train_audio_model.py
  python scripts/product_recommendation.py
  ```
- Ensure models are saved in `models/` directory
- Check that all model files (model, encoder, feature columns) are present

### Import Errors
**Problem:** `ModuleNotFoundError` when running scripts

**Solutions:**
- Install all requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (3.7+ recommended)
- For macOS, you may need: `brew install portaudio` before installing pyaudio

### Audio Processing Issues
**Problem:** Audio files not loading or processing errors

**Solutions:**
- Ensure audio files are in WAV format (or convert them)
- Check file permissions
- Verify audio files are not corrupted
- Sample rate conversion is automatic (target: 16 kHz)

### Image Processing Issues
**Problem:** Images not loading or processing errors

**Solutions:**
- Ensure images are in JPG/JPEG format
- Check image file permissions
- Verify images are not corrupted
- Ensure images contain faces (for best results)

### Low Model Accuracy
**Problem:** Models show low accuracy or poor predictions

**Solutions:**
- Ensure sufficient training data (multiple images/audio per member)
- Check that augmentations are being applied correctly
- Verify feature extraction is working (check feature CSV files)
- Try adjusting model hyperparameters in training scripts

## Contributors

- Phinah
- Sage
- Ayomide
- Carine

---


