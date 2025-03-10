# Audio Classification: Cats vs. Dogs

## Overview
This project classifies audio files of cat and dog sounds using machine learning. It utilizes the Kaggle dataset "mmoreaux/audio-cats-and-dogs" and employs MFCC feature extraction along with an SVM model for classification.

## Dataset
The dataset is downloaded from Kaggle using `kagglehub` and contains:
- **Training Data:** Stored in `/content/cats_dogs/train`
- **Testing Data:** Stored in `/content/cats_dogs`

The dataset consists of `.wav` files for cat and dog sounds.

## Dependencies
Ensure you have the following dependencies installed before running the script:

```bash
pip install kagglehub torchaudio librosa numpy scikit-learn
```

## Project Structure
```
.
├── dataset_download.py  # Script to download dataset
├── train.py             # Training and evaluation script
└── README.md            # Project documentation
```

## Features Extraction
The project extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** from audio files using `librosa`. These extracted features are used to train an **SVM classifier**.

## Training Process
1. Load and preprocess audio files.
2. Extract MFCC features.
3. Normalize features using `StandardScaler`.
4. Train an **SVM (Support Vector Machine)** classifier.

## Testing & Evaluation
- Extract features from test audio files.
- Predict the class using the trained SVM model.
- Evaluate accuracy using `accuracy_score`.

## Running the Project
Execute the script in a Python environment:

```bash
python train.py
```

## Expected Output
- The script prints the dataset paths, extracted features, and accuracy.
- Example:
  ```
  Accuracy: 0.85
  ```

## Troubleshooting
- If you encounter an error processing audio files, ensure the dataset is properly downloaded.
- Some files may be skipped due to format issues.

## Future Improvements
- Use deep learning models like CNNs for better accuracy.
- Augment the dataset with additional audio samples.

## Author
- Manish Bharath

