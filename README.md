# Agri-Insight Nepal: Dashboard 🌾

Unified deep learning analysis tools for agricultural monitoring in Nepal. This Streamlit web application features multi-crop plant disease detection and sequence-based seasonal yield estimation leveraging satellite climate data.

## 🚀 Features

### 1. 🌿 Leaf Disease Detector
* **Gatekeeper Filter:** Validates uploaded leaf images using an **EfficientNet-B0** model to confirm the plant species. Rejects invalid or low-confidence inputs (Threshold: 70%).
* **Specialist Diagnosis:** Feeds verified leaf images into plant-specific **Convolutional Neural Networks (CNN)** to diagnose explicit disease labels or confirm plant health.
* **Coverage:** Supports **14 distinct plant species** (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).

### 2. 📊 Paddy Yield Predictor
* **Real-Time Climate Ingestion:** Fetches real-world historical and live climate metrics (precipitation, mean temperature, relative humidity) via the **Open-Meteo Archive API**.
* **LSTM Time-Series Sequence Forecasting:** Processes data through a 27-week timeline into a 33-dimensional input frame (24-district one-hot encoding vector + 6 normalized physical soil parameters + 3 dynamic weather tracks).
* **NARC Soil Registry:** Integrated with static physical metrics (cultivated area, elevation, pH, organic matter, clay, and sand content) for major agricultural zones across Nepal.

---

## 🛠️ Tech Stack
* **Frontend Dashboard:** [Streamlit](https://streamlit.io/)
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/)
* **Computer Vision Architectures:** `timm` (EfficientNet-B0), Custom PyTorch 2D CNN
* **Time-Series Forecasting:** Custom PyTorch LSTM (`DeeperPaddyYieldLSTM`)
* **Data Sources:** Open-Meteo API, Nepal Agricultural Research Council (NARC)

---

## 📂 Project Structure & Required Weights
To run this application, download and place the required weights files (`.pth`) in your main project folder directory:

```bash
smart_multi_crop_disease_detection/
├── app.py                               # Main Streamlit application file
├── efficientnet_b0_gatekeeper.pth       # Plant species classification weights
├── spatial_paddy_lstm_final.pth         # 33-D Paddy Yield prediction weights
├── Apple_disease_cnn.pth                # Plant-specific disease CNN weights
├── Tomato_disease_cnn.pth               # ...
└── [Other Plant]_disease_cnn.pth        # Additional disease model weights
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ankitroy1st456-lang/smart_multi_crop_disease_detection.git
   cd smart_multi_crop_disease_detection
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv crop_stable
   source crop_stable/bin/activate  # On Windows use: crop_stable\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install streamlit torch torchvision timm numpy requests pillow
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📋 License
This project is open-source and available under the MIT License.

---
*Developed for smart multi-crop disease detection and agricultural monitoring in Nepal.*
