# 🔧 Bearing Fault Detection with Deep Learning

A complete machine learning pipeline for detecting and classifying bearing faults using 
CNN-based spectrogram analysis — trained on the CWRU Bearing Dataset, 
exportable to TFLite for edge deployment.

## 📋 Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |
| Classes   | 6     |
| Model Size| 1.2 MB (TFLite) |

## 🏗️ Project Structure

BearingFaultDetection/<br>
├── data/<br>
│   ├── normal/               # CWRU normal baseline data<br>
│   └── 12k drive end/        # Fault data at 4 load points<br>
├── notebooks/<br>
│   ├── 01_data_exploration.ipynb     # Signal analysis & FFT/STFT visualization<br>
│   ├── 02_feature_extraction.ipynb   # Spectrogram patch extraction<br>
│   └── 03_cnn_training.ipynb         # Model training & TFLite export<br>
├── models/<br>
│   ├── X_patches.npy                 # Input: (480, 65, 185) spectrogram patches<br>
│   ├── y_labels.npy                  # Labels: 6 fault classes<br>
│   ├── features.csv                  # Handcrafted features (RMS, Kurtosis, etc.)<br>
│   └── bearing_fault_model.tflite    # Edge-ready model<br>
└── src/

## 🎯 Fault Classes

| Label | Class              | Description                    |
|-------|--------------------|--------------------------------|
| 0     | Normal             | Healthy bearing                |
| 1     | Inner Race         | Fault on inner race            |
| 2     | Ball               | Fault on rolling element       |
| 3     | Outer Race Center  | OR fault in load zone (6:00)   |
| 4     | Outer Race Opposite| OR fault opposite load (12:00) |
| 5     | Outer Race Ortho   | OR fault 90° offset (3:00)     |

## 🧠 Model Architecture

Input (65×185×1 spectrogram patch)<br>
→ Conv2D(8) + BatchNorm + MaxPool + Dropout(0.3)<br>
→ Conv2D(16) + BatchNorm + MaxPool + Dropout(0.3)<br>
→ Flatten → Dense(32) + Dropout(0.4)<br>
→ Dense(6, softmax)<br>
Parameters: ~50k | Regularization: L2(0.01)<br>

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/BearingFaultDetection.git
cd BearingFaultDetection
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas scikit-learn tensorflow
```

### 2. Data
Download the [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter):
- Drive End, 12kHz, 0.007" fault diameter
- Load points: 0, 1, 2, 3 HP (1797, 1772, 1750, 1730 RPM)

Place `.mat` files in `data/normal/` and `data/12k drive end/`.

### 3. Run Notebooks
Execute in order:<br>
01_data_exploration.ipynb   → Visualize signals & spectrograms<br>
02_feature_extraction.ipynb → Extract & save patches<br>
03_cnn_training.ipynb       → Train model & export TFLite<br>

## ⚙️ TFLite Inference

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/bearing_fault_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()   # [1, 65, 185, 1] float32
output_details = interpreter.get_output_details() # [1, 6] float32

# Run inference
interpreter.set_tensor(input_details[0]['index'], patch)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
```

## 🔑 Key Design Decisions

- **Spectrograms over raw signals** – STFT captures time-frequency structure of fault patterns
- **Normalization after split** – prevents data leakage, critical for honest evaluation  
- **L2 + Dropout** – combats overfitting on small dataset (480 samples)
- **TFLite export** – runs on Raspberry Pi, STM32, ESP32 without TensorFlow

## 📊 Dataset

- **Source:** [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
- **Sampling rate:** 12,000 Hz
- **Window size:** 12,000 samples (1 second)
- **Step size:** 6,000 samples (50% overlap)
- **Total windows:** 480 (80 per class × 6 classes)

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![NumPy](https://img.shields.io/badge/NumPy-scientific-green)
![SciPy](https://img.shields.io/badge/SciPy-signal-lightblue)

## 📈 Next Steps

- [ ] Test with larger fault diameters (0.014", 0.021")
- [ ] Deploy on Raspberry Pi 4 with real-time inference
- [ ] Add Fan End bearing data for broader coverage