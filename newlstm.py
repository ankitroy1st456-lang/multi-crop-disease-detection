# # # import streamlit as st
# # # import torch
# # # import torch.nn as nn
# # # import timm
# # # from torchvision import transforms
# # # from PIL import Image
# # # import os
# # # import numpy as np
# # # import requests

# # # # ============================================================
# # # # CONFIG, MODEL DEFINITIONS & HARDWARE MAPPING
# # # # ============================================================
# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # PLANT_CONFIDENCE_THRESHOLD = 70.0
# # # GATEKEEPER_PATH = "efficientnet_b0_gatekeeper.pth"
# # # NUM_PLANT_CLASSES = 14

# # # # folder-name -> plant species mapping
# # # PLANT_CLASS_TO_IDX = {
# # #     'Apple': 0, 'Blueberry': 1, 'Cherry': 2, 'Corn': 3, 'Grape': 4,
# # #     'Orange': 5, 'Peach': 6, 'Pepper,': 7, 'Potato': 8, 'Raspberry': 9,
# # #     'Soybean': 10, 'Squash': 11, 'Strawberry': 12, 'Tomato': 13
# # # }
# # # IDX_TO_PLANT = {v: k for k, v in PLANT_CLASS_TO_IDX.items()}

# # # DISEASE_CLASSES = {
# # #     "Apple": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Blueberry": ['Blueberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Cherry": ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Corn": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Grape": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Orange": ['Orange___Haunglongbing_(Citrus_greening)', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Peach": ['Peach___Bacterial_spot', 'Peach___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Pepper,": ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Raspberry": ['Raspberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Soybean": ['Soybean___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Squash": ['Squash___Powdery_mildew', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Strawberry": ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# # #     "Tomato": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
# # # }

# # # def disease_model_filename(plant: str) -> str:
# # #     return f"{plant}_disease_cnn.pth"

# # # class PlantDiseaseCNN(nn.Module):
# # #     def __init__(self, num_classes=10):
# # #         super(PlantDiseaseCNN, self).__init__()
# # #         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# # #         self.relu1 = nn.ReLU()
# # #         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
# # #         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
# # #         self.relu2 = nn.ReLU()
# # #         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# # #         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
# # #         self.relu3 = nn.ReLU()
# # #         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
# # #         self.fc1 = nn.Linear(64 * 28 * 28, 256)
# # #         self.relu4 = nn.ReLU()
# # #         self.fc2 = nn.Linear(256, num_classes)

# # #     def forward(self, x):
# # #         x = self.pool1(self.relu1(self.conv1(x)))
# # #         x = self.pool2(self.relu2(self.conv2(x)))
# # #         x = self.pool3(self.relu3(self.conv3(x)))
# # #         x = torch.flatten(x, 1)
# # #         x = self.relu4(self.fc1(x))
# # #         x = self.fc2(x)
# # #         return x

# # # class DeeperPaddyYieldLSTM(nn.Module):
# # #     # Change default input_dim constraint from 3 to 33
# # #     def __init__(self, input_dim=33, hidden_dim=128, num_layers=2, output_dim=1):
# # #         super(DeeperPaddyYieldLSTM, self).__init__()
# # #         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
# # #         self.dropout = nn.Dropout(0.3)
# # #         self.fc = nn.Linear(hidden_dim, output_dim)

# # #     def forward(self, x):
# # #         # Input tensor shape will be: [1, 27, 33]
# # #         out, _ = self.lstm(x)
# # #         return self.fc(self.dropout(out[:, -1, :]))


# # # transform = transforms.Compose([
# # #     transforms.Resize((224, 224)),
# # #     transforms.ToTensor(),
# # #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # # ])

# # # @st.cache_resource
# # # def load_gatekeeper():
# # #     model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_PLANT_CLASSES)
# # #     num_features = model.classifier.in_features
# # #     model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_features, NUM_PLANT_CLASSES))
# # #     model.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=DEVICE))
# # #     return model.to(DEVICE).eval()

# # # @st.cache_resource
# # # def load_disease_model(plant: str):
# # #     filename = disease_model_filename(plant)
# # #     if not os.path.exists(filename):
# # #         return None, filename
# # #     model = PlantDiseaseCNN(num_classes=len(DISEASE_CLASSES[plant]))
# # #     model.load_state_dict(torch.load(filename, map_location=DEVICE))
# # #     return model.to(DEVICE).eval(), filename


# # # @st.cache_resource
# # # def load_yield_model():
# # #     # Update to point to your new Version 2.0 weights file
# # #     model_path = "spatial_paddy_lstm_final.pth"
# # #     if not os.path.exists(model_path):
# # #         return None
    
# # #     # Initialize the architecture frame with input_dim=33
# # #     model = DeeperPaddyYieldLSTM(input_dim=33, hidden_dim=128, num_layers=2, output_dim=1)
# # #     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# # #     return model.to(DEVICE).eval()


# # # # ============================================================
# # # # STREAMLIT MAIN VISUAL APP LAYOUT
# # # # ============================================================
# # # st.set_page_config(page_title="Agri-Insight Nepal", page_icon="🌾")
# # # st.title("🌾 Agri-Insight Nepal: Dashboard")
# # # st.write("Unified deep learning analysis tools for agricultural monitoring in Nepal.")

# # # tab1, tab2 = st.tabs(["🌿 Leaf Disease Detector", "📊 Paddy Yield Predictor"])

# # # # ============================================================
# # # # TAB 1: DISEASE DETECTOR VIEW (Original Code Preserved)
# # # # ============================================================
# # # with tab1:
# # #     st.header("Leaf Health Diagnostics")
# # #     uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# # #     if uploaded_file is not None:
# # #         image = Image.open(uploaded_file).convert("RGB")
# # #         st.image(image, caption="Uploaded Image", width='stretch')

# # #         if st.button("Predict"):
# # #             img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# # #             with st.spinner("Identifying plant species..."):
# # #                 gatekeeper = load_gatekeeper()
# # #                 with torch.no_grad():
# # #                     plant_logits = gatekeeper(img_tensor)
# # #                     plant_probs = torch.softmax(plant_logits, dim=1)[0]
# # #                     plant_idx = torch.argmax(plant_probs).item()
# # #                     plant_name = IDX_TO_PLANT[plant_idx]
# # #                     plant_confidence = plant_probs[plant_idx].item() * 100

# # #             if plant_confidence < PLANT_CONFIDENCE_THRESHOLD:
# # #                 st.error(f"⚠️ **Image Rejected** (Plant Recognition Confidence: {plant_confidence:.2f}%). The model is not sure what kind of leaf this is.")
# # #             else:
# # #                 st.success(f"**Step 1 — Detected Plant: {plant_name}**  (confidence: {plant_confidence:.2f}%)")

# # #                 with st.spinner(f"Loading {plant_name} disease model and predicting..."):
# # #                     disease_model, filename = load_disease_model(plant_name)

# # #                 if disease_model is None:
# # #                     st.error(f"Disease model file `{filename}` not found. Place it in the same folder as this app.")
# # #                 else:
# # #                     with torch.no_grad():
# # #                         disease_logits = disease_model(img_tensor)
# # #                         disease_probs = torch.softmax(disease_logits, dim=1)[0]
# # #                         disease_idx = torch.argmax(disease_probs).item()
# # #                         disease_label = DISEASE_CLASSES[plant_name][disease_idx]
# # #                         disease_confidence = disease_probs[disease_idx].item() * 100

# # #                     if disease_confidence < PLANT_CONFIDENCE_THRESHOLD:
# # #                         st.warning(f"⚠️ **Prediction Rejected** (Confidence: {disease_confidence:.2f}%). The Specialist model is too uncertain.")
# # #                     else:
# # #                         display_disease = disease_label.split("___", 1)[-1].replace("_", " ")
# # #                         st.success(f"**Step 2 — Diagnosis: {display_disease}**  (confidence: {disease_confidence:.2f}%)")

# # #                     top_k = min(3, len(DISEASE_CLASSES[plant_name]))
# # #                     topk = torch.topk(disease_probs, top_k)
# # #                     st.write("Top predictions:")
# # #                     for prob, idx in zip(topk.values, topk.indices):
# # #                         label = DISEASE_CLASSES[plant_name][idx.item()].split("___", 1)[-1].replace("_", " ")
# # #                         st.write(f"- {label}: {prob.item()*100:.2f}%")

# # #     st.divider()
# # #     with st.expander("Required disease model files"):
# # #         st.write(f"- `{GATEKEEPER_PATH}` (gatekeeper)")
# # #         for plant in DISEASE_CLASSES:
# # #             st.write(f"- `{disease_model_filename(plant)}` ({len(DISEASE_CLASSES[plant])} classes)")

# # # # ============================================================
# # # # TAB 2: YIELD PREDICTOR VIEW (Live Weather Mode - Fixed)
# # # # ============================================================
# # # with tab2:
# # #     st.header("🌾 Real-Time Seasonal Yield Estimation")
# # #     st.write("Fetches live climate data for the selected district to predict expected Paddy yield.")

# # #     # Expanded district coordinate lookup map covering Nepal's major agricultural zones
# # #     district_coords = {
# # #         "Jhapa": {"lat": 26.63, "lon": 87.90},
# # #         "Morang": {"lat": 26.65, "lon": 87.42},
# # #         "Sunsari": {"lat": 26.62, "lon": 87.15},
# # #         "Chitwan": {"lat": 27.58, "lon": 84.49},
# # #         "Bara": {"lat": 27.05, "lon": 85.02},
# # #         "Parsa": {"lat": 27.15, "lon": 84.88},
# # #         "Rautahat": {"lat": 26.84, "lon": 85.26},
# # #         "Dhanusha": {"lat": 26.78, "lon": 85.97},
# # #         "Siraha": {"lat": 26.65, "lon": 86.21},
# # #         "Saptari": {"lat": 26.54, "lon": 86.74},
# # #         "Rupandehi": {"lat": 27.53, "lon": 83.45},
# # #         "Kapilvastu": {"lat": 27.53, "lon": 82.95},
# # #         "Nawalparasi": {"lat": 27.53, "lon": 83.98},
# # #         "Banke": {"lat": 28.13, "lon": 81.65},
# # #         "Bardiya": {"lat": 28.32, "lon": 81.36},
# # #         "Kailali": {"lat": 28.78, "lon": 80.86},
# # #         "Kanchanpur": {"lat": 28.92, "lon": 80.33},
# # #         "Dang": {"lat": 28.01, "lon": 82.31},
# # #         "Kaski": {"lat": 28.25, "lon": 83.98},
# # #         "Kavre": {"lat": 27.52, "lon": 85.55},
# # #         "Bhaktapur": {"lat": 27.67, "lon": 85.43},
# # #         "Lalitpur": {"lat": 27.60, "lon": 85.33},
# # #         "Kathmandu": {"lat": 27.71, "lon": 85.32},
# # #         "Palpa": {"lat": 27.86, "lon": 83.55}
# # #     }

# # #     # Static Soil and Physical Registry (NARC derived metrics)
# # #     SPATIAL_REGISTRY = {
# # #         "Jhapa":       {"idx": 0,  "area": 88500.0, "elev": 125.0, "ph": 6.0282, "clay": 18.7874, "sand": 44.3958, "org": 3.7338},
# # #         "Bara":        {"idx": 4,  "area": 54680.0, "elev": 95.0,  "ph": 7.2795, "clay": 21.2574, "sand": 35.5916, "org": 1.9240},
# # #         "Palpa":       {"idx": 23, "area": 9430.0,  "elev": 360.0, "ph": 6.5733, "clay": 13.7497, "sand": 42.7099, "org": 2.9248},
# # #         "Siraha":      {"idx": 8,  "area": 40000.0, "elev": 105.0, "ph": 6.2000, "clay": 25.0000, "sand": 43.0000, "org": 1.6000},
# # #         "Parsa":       {"idx": 5,  "area": 45000.0, "elev": 115.0, "ph": 7.1500, "clay": 22.0000, "sand": 36.0000, "org": 1.8000},
# # #         "Morang":      {"idx": 1,  "area": 81000.0, "elev": 130.0, "ph": 5.6000, "clay": 24.0000, "sand": 42.0000, "org": 2.0000},
# # #         "Sunsari":     {"idx": 2,  "area": 53000.0, "elev": 140.0, "ph": 5.8000, "clay": 22.0000, "sand": 45.0000, "org": 1.9000},
# # #         "Chitwan":     {"idx": 3,  "area": 29000.0, "elev": 208.0, "ph": 6.0000, "clay": 20.0000, "sand": 40.0000, "org": 2.4000},
# # #         "Rautahat":    {"idx": 6,  "area": 39000.0, "elev": 85.0,  "ph": 6.1000, "clay": 27.0000, "sand": 38.0000, "org": 1.6000},
# # #         "Dhanusha":    {"idx": 7,  "area": 46000.0, "elev": 92.0,  "ph": 6.4000, "clay": 26.0000, "sand": 41.0000, "org": 1.5000},
# # #         "Saptari":     {"idx": 9,  "area": 41000.0, "elev": 88.0,  "ph": 6.0000, "clay": 26.0000, "sand": 42.0000, "org": 1.7000},
# # #         "Rupandehi":   {"idx": 10, "area": 68000.0, "elev": 120.0, "ph": 6.8000, "clay": 30.0000, "sand": 30.0000, "org": 2.2000},
# # #         "Kapilvastu":  {"idx": 11, "area": 71000.0, "elev": 110.0, "ph": 6.7000, "clay": 28.0000, "sand": 34.0000, "org": 2.0000},
# # #         "Nawalparasi": {"idx": 12, "area": 42000.0, "elev": 150.0, "ph": 6.4000, "clay": 24.0000, "sand": 38.0000, "org": 2.1000},
# # #         "Banke":       {"idx": 13, "area": 36000.0, "elev": 145.0, "ph": 7.0000, "clay": 22.0000, "sand": 48.0000, "org": 1.8000},
# # #         "Bardiya":     {"idx": 14, "area": 52000.0, "elev": 155.0, "ph": 6.9000, "clay": 23.0000, "sand": 46.0000, "org": 1.9000},
# # #         "Kailali":     {"idx": 15, "area": 72000.0, "elev": 190.0, "ph": 6.5000, "clay": 25.0000, "sand": 40.0000, "org": 2.0000},
# # #         "Kanchanpur":  {"idx": 16, "area": 48000.0, "elev": 210.0, "ph": 6.3000, "clay": 24.0000, "sand": 41.0000, "org": 2.2000},
# # #         "Dang":        {"idx": 17, "area": 37000.0, "elev": 650.0, "ph": 6.6000, "clay": 21.0000, "sand": 44.0000, "org": 2.3000},
# # #         "Kaski":       {"idx": 18, "area": 25000.0, "elev": 890.0, "ph": 5.2000, "clay": 18.0000, "sand": 42.0000, "org": 3.2000},
# # #         "Kavre":       {"idx": 19, "area": 11000.0, "elev": 1450., "ph": 5.1000, "clay": 26.0000, "sand": 34.0000, "org": 2.8000},
# # #         "Bhaktapur":   {"idx": 20, "area": 4000.0,  "elev": 1330., "ph": 5.4000, "clay": 28.0000, "sand": 30.0000, "org": 2.6000},
# # #         "Lalitpur":    {"idx": 21, "area": 9000.0,  "elev": 1350., "ph": 5.3000, "clay": 27.0000, "sand": 32.0000, "org": 2.7000},
# # #         "Kathmandu":   {"idx": 22, "area": 12000.0, "elev": 1340., "ph": 5.5000, "clay": 25.0000, "sand": 35.0000, "org": 2.5000}
# # #     }

# # #     district_choice = st.selectbox("Select target Nepal District", list(district_coords.keys()))
# # #     crop_choice = st.selectbox("Select Target Crop", ["Paddy (Rice)", "Maize (Coming Soon)", "Wheat (Coming Soon)"])

# # #     if st.button("Calculate Expected Yield"):
# # #         with st.spinner("Connecting to global satellite feeds to pull real-world seasonal data..."):
# # #             coords = district_coords[district_choice]
# # #             base_url = "https://archive-api.open-meteo.com/v1/archive"

# # #             query_params = {
# # #                 "latitude": coords["lat"],
# # #                 "longitude": coords["lon"],
# # #                 "start_date": "2024-06-01",
# # #                 "end_date": "2024-11-30",
# # #                 "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
# # #                 "timezone": "auto"
# # #             }

# # #             try:
# # #                 headers = {"Accept": "application/json"}
# # #                 response = requests.get(base_url, params=query_params, headers=headers, timeout=15)

# # #                 if response.status_code != 200:
# # #                     st.error(f"Weather Server returned an unexpected error status: Code {response.status_code}. Please try again later.")
# # #                 else:
# # #                     data = response.json()

# # #                     if "daily" in data:
# # #                         raw_rain = data["daily"]["precipitation_sum"]
# # #                         raw_temp = data["daily"]["temperature_2m_mean"]
# # #                         raw_hum = data["daily"]["relative_humidity_2m_mean"]

# # #                         rain_arr = np.nan_to_num(np.array(raw_rain, dtype=np.float32))
# # #                         temp_arr = np.nan_to_num(np.array(raw_temp, dtype=np.float32))
# # #                         hum_arr = np.nan_to_num(np.array(raw_hum, dtype=np.float32))

# # #                         weeks_input = []
# # #                         for i in range(27):
# # #                             start_idx = i * 7
# # #                             end_idx = start_idx + 7
# # #                             weeks_input.append([
# # #                                 float(np.sum(rain_arr[start_idx:end_idx])),
# # #                                 float(np.mean(temp_arr[start_idx:end_idx])),
# # #                                 float(np.mean(hum_arr[start_idx:end_idx]))
# # #                             ])

# # #                                                 # 1. Extract static physical features for the selection
# # #                         meta = SPATIAL_REGISTRY[district_choice]
                        
# # #                         # Build the 24-dimensional One-Hot row vector component
# # #                         one_hot = np.zeros(24, dtype=np.float32)
# # #                         one_hot[meta["idx"]] = 1.0

# # #                         # Normalize the 6 static features exactly like training bounds
# # #                         n_area = (meta["area"] - 4000.0) / (92000.0 - 4000.0)
# # #                         n_elev = (meta["elev"] - 85.0) / (1450.0 - 85.0)
# # #                         n_ph   = (meta["ph"] - 4.5) / (8.0 - 4.5)
# # #                         n_clay = (meta["clay"] - 5.0) / (35.0 - 5.0)
# # #                         n_sand = (meta["sand"] - 20.0) / (60.0 - 20.0)
# # #                         n_org  = (meta["org"] - 0.5) / (5.0 - 0.5)
# # #                         static_30d = np.concatenate([one_hot, [n_area, n_elev, n_ph, n_clay, n_sand, n_org]])

# # #                         # 2. Step through weeks, normalize weather, and stitch to reach 33 inputs
# # #                         scaled_sequence = []
# # #                         for w in weeks_input:
# # #                             rain, temp, hum = w[0], w[1], w[2]

# # #                             norm_rain = (rain - 0.0) / (365.73 - 0.0)
# # #                             norm_temp = (temp - (-14.39)) / (36.926 - (-14.39))
# # #                             norm_hum = (hum - 11.41) / (95.437 - 11.41)

# # #                             weather_3d = np.array([
# # #                                 float(np.clip(norm_rain, 0.0, 1.0)),
# # #                                 float(np.clip(norm_temp, 0.0, 1.0)),
# # #                                 float(np.clip(norm_hum, 0.0, 1.0))
# # #                             ], dtype=np.float32)

# # #                             # Concatenate 30 static slots + 3 weather slots = 33 dimensions
# # #                             week_33d = np.concatenate([static_30d, weather_3d])
# # #                             scaled_sequence.append(week_33d)

# # #                         # 3. Create input tensor array of shape [1, 27, 33]
# # #                         input_tensor = torch.tensor([scaled_sequence], dtype=torch.float32).to(DEVICE)

# # #                         yield_model = load_yield_model()

# # #                         if yield_model is None:
# # #                             st.error("⚠️ Upgraded model file `spatial_paddy_lstm_final.pth` not found inside your folder directory.")
# # #                         else:
# # #                             with torch.no_grad():
# # #                                 raw_pred = yield_model(input_tensor).cpu().item()

# # #                             # Reverse Min/Max training scaling bounds to output real Metric Tons / Hectare
# # #                             final_yield = float(raw_pred) * (4.751 - 0.33) + 0.33

# # #                             st.write("---")
# # #                             st.subheader("📈 Yield Prediction Result")

# # #                             r1, r2 = st.columns([1, 1])
# # #                             r1.metric(label="Projected Harvest Yield", value=f"{final_yield:.2f} t/ha")
# # #                             avg_rain_wk = float(np.mean([w[0] for w in weeks_input]))
# # #                             avg_temp_wk = float(np.mean([w[1] for w in weeks_input]))
# # #                             r2.metric(label="Avg Weekly Rainfall", value=f"{avg_rain_wk:.1f} mm")

# # #                             st.caption(f"Calculated for {district_choice} using updated 33-D physical soil parameters.")

# # #                             st.write("**Soil & Site Profile Used In This Prediction**")
# # #                             s1, s2, s3 = st.columns(3)
# # #                             s1.metric(label="⛰️ Elevation", value=f"{meta['elev']:.0f} m")
# # #                             s2.metric(label="🗺️ Cultivated Area", value=f"{meta['area']:,} Ha")
# # #                             s3.metric(label="🧪 Topsoil pH", value=f"{meta['ph']:.2f}")

# # #                             s4, s5, s6 = st.columns(3)
# # #                             s4.metric(label="🌿 Organic Matter", value=f"{meta['org']:.2f} %")
# # #                             s5.metric(label="🧱 Clay Content", value=f"{meta['clay']:.1f} %")
# # #                             s6.metric(label="⏳ Sand Content", value=f"{meta['sand']:.1f} %")

# # #                             st.write("**Soil Texture Breakdown**")
# # #                             st.caption(f"Clay (moisture holding): {meta['clay']:.1f}%")
# # #                             st.progress(min(max(int(meta['clay'] * 2), 0), 100))
# # #                             st.caption(f"Sand (drainage capacity): {meta['sand']:.1f}%")
# # #                             st.progress(min(max(int(meta['sand']), 0), 100))

# # #                             st.write("**Seasonal Weather Summary (used in prediction)**")
# # #                             w1, w2, w3 = st.columns(3)
# # #                             w1.metric(label="🌧️ Avg Weekly Rain", value=f"{avg_rain_wk:.1f} mm")
# # #                             w2.metric(label="🌡️ Avg Temperature", value=f"{avg_temp_wk:.1f} °C")
# # #                             w3.metric(label="📅 Weeks Analyzed", value="27")

# # #                             with st.expander("How this yield number was calculated"):
# # #                                 st.write(
# # #                                     "The model combines a 24-district one-hot location vector, "
# # #                                     "6 normalized static soil/site features (area, elevation, pH, clay, "
# # #                                     "sand, organic matter), and 3 weekly weather features "
# # #                                     "(rainfall, temperature, humidity) across 27 weeks into a "
# # #                                     "33-dimensional sequence, then feeds it through the LSTM to "
# # #                                     "predict yield in metric tons per hectare."
# # #                                 )
# # #                             st.write("---")
# # #                     else:
# # #                         st.error("Could not decode local satellite streaming response arrays. Please try again.")
# # #             except Exception as e:
# # #                 st.error(f"Network Error: {str(e)}")

# # import streamlit as st
# # import torch
# # import torch.nn as nn
# # import timm
# # from torchvision import transforms
# # from PIL import Image
# # import os
# # import numpy as np
# # import requests

# # # ============================================================
# # # CONFIG, MODEL DEFINITIONS & HARDWARE MAPPING
# # # ============================================================
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # PLANT_CONFIDENCE_THRESHOLD = 70.0
# # GATEKEEPER_PATH = "efficientnet_b0_gatekeeper.pth"
# # NUM_PLANT_CLASSES = 14

# # # folder-name -> plant species mapping
# # PLANT_CLASS_TO_IDX = {
# #     'Apple': 0, 'Blueberry': 1, 'Cherry': 2, 'Corn': 3, 'Grape': 4,
# #     'Orange': 5, 'Peach': 6, 'Pepper,': 7, 'Potato': 8, 'Raspberry': 9,
# #     'Soybean': 10, 'Squash': 11, 'Strawberry': 12, 'Tomato': 13
# # }
# # IDX_TO_PLANT = {v: k for k, v in PLANT_CLASS_TO_IDX.items()}

# # DISEASE_CLASSES = {
# #     "Apple": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Blueberry": ['Blueberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Cherry": ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Corn": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Grape": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Orange": ['Orange___Haunglongbing_(Citrus_greening)', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Peach": ['Peach___Bacterial_spot', 'Peach___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Pepper,": ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Raspberry": ['Raspberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Soybean": ['Soybean___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Squash": ['Squash___Powdery_mildew', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Strawberry": ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
# #     "Tomato": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
# # }

# # def disease_model_filename(plant: str) -> str:
# #     return f"{plant}_disease_cnn.pth"

# # class PlantDiseaseCNN(nn.Module):
# #     def __init__(self, num_classes=10):
# #         super(PlantDiseaseCNN, self).__init__()
# #         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# #         self.relu1 = nn.ReLU()
# #         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
# #         self.relu2 = nn.ReLU()
# #         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
# #         self.relu3 = nn.ReLU()
# #         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.fc1 = nn.Linear(64 * 28 * 28, 256)
# #         self.relu4 = nn.ReLU()
# #         self.fc2 = nn.Linear(256, num_classes)

# #     def forward(self, x):
# #         x = self.pool1(self.relu1(self.conv1(x)))
# #         x = self.pool2(self.relu2(self.conv2(x)))
# #         x = self.pool3(self.relu3(self.conv3(x)))
# #         x = torch.flatten(x, 1)
# #         x = self.relu4(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # class DeeperPaddyYieldLSTM(nn.Module):
# #     # Change default input_dim constraint from 3 to 33
# #     def __init__(self, input_dim=33, hidden_dim=128, num_layers=2, output_dim=1):
# #         super(DeeperPaddyYieldLSTM, self).__init__()
# #         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
# #         self.dropout = nn.Dropout(0.3)
# #         self.fc = nn.Linear(hidden_dim, output_dim)

# #     def forward(self, x):
# #         # Input tensor shape will be: [1, 27, 33]
# #         out, _ = self.lstm(x)
# #         return self.fc(self.dropout(out[:, -1, :]))


# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # ])

# # @st.cache_resource
# # def load_gatekeeper():
# #     model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_PLANT_CLASSES)
# #     num_features = model.classifier.in_features
# #     model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_features, NUM_PLANT_CLASSES))
# #     model.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=DEVICE))
# #     return model.to(DEVICE).eval()

# # @st.cache_resource
# # def load_disease_model(plant: str):
# #     filename = disease_model_filename(plant)
# #     if not os.path.exists(filename):
# #         return None, filename
# #     model = PlantDiseaseCNN(num_classes=len(DISEASE_CLASSES[plant]))
# #     checkpoint = torch.load(filename, map_location=DEVICE)
# #     # Some checkpoints are saved as a raw state_dict, others are wrapped
# #     # in a dict like {"format": ..., "state_dict": {...}}. Unwrap if needed.
# #     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
# #         checkpoint = checkpoint["state_dict"]
# #     model.load_state_dict(checkpoint)
# #     return model.to(DEVICE).eval(), filename


# # @st.cache_resource
# # def load_yield_model():
# #     # Update to point to your new Version 2.0 weights file
# #     model_path = "spatial_paddy_lstm_final.pth"
# #     if not os.path.exists(model_path):
# #         return None
    
# #     # Initialize the architecture frame with input_dim=33
# #     model = DeeperPaddyYieldLSTM(input_dim=33, hidden_dim=128, num_layers=2, output_dim=1)
# #     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# #     return model.to(DEVICE).eval()


# # # ============================================================
# # # STREAMLIT MAIN VISUAL APP LAYOUT
# # # ============================================================
# # st.set_page_config(page_title="Agri-Insight Nepal", page_icon="🌾")
# # st.title("🌾 Agri-Insight Nepal: Dashboard")
# # st.write("Unified deep learning analysis tools for agricultural monitoring in Nepal.")

# # tab1, tab2 = st.tabs(["🌿 Leaf Disease Detector", "📊 Paddy Yield Predictor"])

# # # ============================================================
# # # TAB 1: DISEASE DETECTOR VIEW (Original Code Preserved)
# # # ============================================================
# # with tab1:
# #     st.header("Leaf Health Diagnostics")
# #     uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# #     if uploaded_file is not None:
# #         image = Image.open(uploaded_file).convert("RGB")
# #         st.image(image, caption="Uploaded Image", width='stretch')

# #         if st.button("Predict"):
# #             img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# #             with st.spinner("Identifying plant species..."):
# #                 gatekeeper = load_gatekeeper()
# #                 with torch.no_grad():
# #                     plant_logits = gatekeeper(img_tensor)
# #                     plant_probs = torch.softmax(plant_logits, dim=1)[0]
# #                     plant_idx = torch.argmax(plant_probs).item()
# #                     plant_name = IDX_TO_PLANT[plant_idx]
# #                     plant_confidence = plant_probs[plant_idx].item() * 100

# #             if plant_confidence < PLANT_CONFIDENCE_THRESHOLD:
# #                 st.error(f"⚠️ **Image Rejected** (Plant Recognition Confidence: {plant_confidence:.2f}%). The model is not sure what kind of leaf this is.")
# #             else:
# #                 st.success(f"**Step 1 — Detected Plant: {plant_name}**  (confidence: {plant_confidence:.2f}%)")

# #                 with st.spinner(f"Loading {plant_name} disease model and predicting..."):
# #                     disease_model, filename = load_disease_model(plant_name)

# #                 if disease_model is None:
# #                     st.error(f"Disease model file `{filename}` not found. Place it in the same folder as this app.")
# #                 else:
# #                     with torch.no_grad():
# #                         disease_logits = disease_model(img_tensor)
# #                         disease_probs = torch.softmax(disease_logits, dim=1)[0]
# #                         disease_idx = torch.argmax(disease_probs).item()
# #                         disease_label = DISEASE_CLASSES[plant_name][disease_idx]
# #                         disease_confidence = disease_probs[disease_idx].item() * 100

# #                     if disease_confidence < PLANT_CONFIDENCE_THRESHOLD:
# #                         st.warning(f"⚠️ **Prediction Rejected** (Confidence: {disease_confidence:.2f}%). The Specialist model is too uncertain.")
# #                     else:
# #                         display_disease = disease_label.split("___", 1)[-1].replace("_", " ")
# #                         st.success(f"**Step 2 — Diagnosis: {display_disease}**  (confidence: {disease_confidence:.2f}%)")

# #                     top_k = min(3, len(DISEASE_CLASSES[plant_name]))
# #                     topk = torch.topk(disease_probs, top_k)
# #                     st.write("Top predictions:")
# #                     for prob, idx in zip(topk.values, topk.indices):
# #                         label = DISEASE_CLASSES[plant_name][idx.item()].split("___", 1)[-1].replace("_", " ")
# #                         st.write(f"- {label}: {prob.item()*100:.2f}%")

# #     st.divider()
# #     with st.expander("Required disease model files"):
# #         st.write(f"- `{GATEKEEPER_PATH}` (gatekeeper)")
# #         for plant in DISEASE_CLASSES:
# #             st.write(f"- `{disease_model_filename(plant)}` ({len(DISEASE_CLASSES[plant])} classes)")

# # # ============================================================
# # # TAB 2: YIELD PREDICTOR VIEW (Live Weather Mode - Fixed)
# # # ============================================================
# # with tab2:
# #     st.header("🌾 Real-Time Seasonal Yield Estimation")
# #     st.write("Fetches live climate data for the selected district to predict expected Paddy yield.")

# #     # Expanded district coordinate lookup map covering Nepal's major agricultural zones
# #     district_coords = {
# #         "Jhapa": {"lat": 26.63, "lon": 87.90},
# #         "Morang": {"lat": 26.65, "lon": 87.42},
# #         "Sunsari": {"lat": 26.62, "lon": 87.15},
# #         "Chitwan": {"lat": 27.58, "lon": 84.49},
# #         "Bara": {"lat": 27.05, "lon": 85.02},
# #         "Parsa": {"lat": 27.15, "lon": 84.88},
# #         "Rautahat": {"lat": 26.84, "lon": 85.26},
# #         "Dhanusha": {"lat": 26.78, "lon": 85.97},
# #         "Siraha": {"lat": 26.65, "lon": 86.21},
# #         "Saptari": {"lat": 26.54, "lon": 86.74},
# #         "Rupandehi": {"lat": 27.53, "lon": 83.45},
# #         "Kapilvastu": {"lat": 27.53, "lon": 82.95},
# #         "Nawalparasi": {"lat": 27.53, "lon": 83.98},
# #         "Banke": {"lat": 28.13, "lon": 81.65},
# #         "Bardiya": {"lat": 28.32, "lon": 81.36},
# #         "Kailali": {"lat": 28.78, "lon": 80.86},
# #         "Kanchanpur": {"lat": 28.92, "lon": 80.33},
# #         "Dang": {"lat": 28.01, "lon": 82.31},
# #         "Kaski": {"lat": 28.25, "lon": 83.98},
# #         "Kavre": {"lat": 27.52, "lon": 85.55},
# #         "Bhaktapur": {"lat": 27.67, "lon": 85.43},
# #         "Lalitpur": {"lat": 27.60, "lon": 85.33},
# #         "Kathmandu": {"lat": 27.71, "lon": 85.32},
# #         "Palpa": {"lat": 27.86, "lon": 83.55}
# #     }

# #     # Static Soil and Physical Registry (NARC derived metrics)
# #     SPATIAL_REGISTRY = {
# #         "Jhapa":       {"idx": 0,  "area": 88500.0, "elev": 125.0, "ph": 6.0282, "clay": 18.7874, "sand": 44.3958, "org": 3.7338},
# #         "Bara":        {"idx": 4,  "area": 54680.0, "elev": 95.0,  "ph": 7.2795, "clay": 21.2574, "sand": 35.5916, "org": 1.9240},
# #         "Palpa":       {"idx": 23, "area": 9430.0,  "elev": 360.0, "ph": 6.5733, "clay": 13.7497, "sand": 42.7099, "org": 2.9248},
# #         "Siraha":      {"idx": 8,  "area": 40000.0, "elev": 105.0, "ph": 6.2000, "clay": 25.0000, "sand": 43.0000, "org": 1.6000},
# #         "Parsa":       {"idx": 5,  "area": 45000.0, "elev": 115.0, "ph": 7.1500, "clay": 22.0000, "sand": 36.0000, "org": 1.8000},
# #         "Morang":      {"idx": 1,  "area": 81000.0, "elev": 130.0, "ph": 5.6000, "clay": 24.0000, "sand": 42.0000, "org": 2.0000},
# #         "Sunsari":     {"idx": 2,  "area": 53000.0, "elev": 140.0, "ph": 5.8000, "clay": 22.0000, "sand": 45.0000, "org": 1.9000},
# #         "Chitwan":     {"idx": 3,  "area": 29000.0, "elev": 208.0, "ph": 6.0000, "clay": 20.0000, "sand": 40.0000, "org": 2.4000},
# #         "Rautahat":    {"idx": 6,  "area": 39000.0, "elev": 85.0,  "ph": 6.1000, "clay": 27.0000, "sand": 38.0000, "org": 1.6000},
# #         "Dhanusha":    {"idx": 7,  "area": 46000.0, "elev": 92.0,  "ph": 6.4000, "clay": 26.0000, "sand": 41.0000, "org": 1.5000},
# #         "Saptari":     {"idx": 9,  "area": 41000.0, "elev": 88.0,  "ph": 6.0000, "clay": 26.0000, "sand": 42.0000, "org": 1.7000},
# #         "Rupandehi":   {"idx": 10, "area": 68000.0, "elev": 120.0, "ph": 6.8000, "clay": 30.0000, "sand": 30.0000, "org": 2.2000},
# #         "Kapilvastu":  {"idx": 11, "area": 71000.0, "elev": 110.0, "ph": 6.7000, "clay": 28.0000, "sand": 34.0000, "org": 2.0000},
# #         "Nawalparasi": {"idx": 12, "area": 42000.0, "elev": 150.0, "ph": 6.4000, "clay": 24.0000, "sand": 38.0000, "org": 2.1000},
# #         "Banke":       {"idx": 13, "area": 36000.0, "elev": 145.0, "ph": 7.0000, "clay": 22.0000, "sand": 48.0000, "org": 1.8000},
# #         "Bardiya":     {"idx": 14, "area": 52000.0, "elev": 155.0, "ph": 6.9000, "clay": 23.0000, "sand": 46.0000, "org": 1.9000},
# #         "Kailali":     {"idx": 15, "area": 72000.0, "elev": 190.0, "ph": 6.5000, "clay": 25.0000, "sand": 40.0000, "org": 2.0000},
# #         "Kanchanpur":  {"idx": 16, "area": 48000.0, "elev": 210.0, "ph": 6.3000, "clay": 24.0000, "sand": 41.0000, "org": 2.2000},
# #         "Dang":        {"idx": 17, "area": 37000.0, "elev": 650.0, "ph": 6.6000, "clay": 21.0000, "sand": 44.0000, "org": 2.3000},
# #         "Kaski":       {"idx": 18, "area": 25000.0, "elev": 890.0, "ph": 5.2000, "clay": 18.0000, "sand": 42.0000, "org": 3.2000},
# #         "Kavre":       {"idx": 19, "area": 11000.0, "elev": 1450., "ph": 5.1000, "clay": 26.0000, "sand": 34.0000, "org": 2.8000},
# #         "Bhaktapur":   {"idx": 20, "area": 4000.0,  "elev": 1330., "ph": 5.4000, "clay": 28.0000, "sand": 30.0000, "org": 2.6000},
# #         "Lalitpur":    {"idx": 21, "area": 9000.0,  "elev": 1350., "ph": 5.3000, "clay": 27.0000, "sand": 32.0000, "org": 2.7000},
# #         "Kathmandu":   {"idx": 22, "area": 12000.0, "elev": 1340., "ph": 5.5000, "clay": 25.0000, "sand": 35.0000, "org": 2.5000}
# #     }

# #     district_choice = st.selectbox("Select target Nepal District", list(district_coords.keys()))
# #     crop_choice = st.selectbox("Select Target Crop", ["Paddy (Rice)", "Maize (Coming Soon)", "Wheat (Coming Soon)"])

# #     if st.button("Calculate Expected Yield"):
# #         with st.spinner("Connecting to global satellite feeds to pull real-world seasonal data..."):
# #             coords = district_coords[district_choice]
# #             base_url = "https://archive-api.open-meteo.com/v1/archive"

# #             query_params = {
# #                 "latitude": coords["lat"],
# #                 "longitude": coords["lon"],
# #                 "start_date": "2024-06-01",
# #                 "end_date": "2024-11-30",
# #                 "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
# #                 "timezone": "auto"
# #             }

# #             try:
# #                 headers = {"Accept": "application/json"}
# #                 response = requests.get(base_url, params=query_params, headers=headers, timeout=15)

# #                 if response.status_code != 200:
# #                     st.error(f"Weather Server returned an unexpected error status: Code {response.status_code}. Please try again later.")
# #                 else:
# #                     data = response.json()

# #                     if "daily" in data:
# #                         raw_rain = data["daily"]["precipitation_sum"]
# #                         raw_temp = data["daily"]["temperature_2m_mean"]
# #                         raw_hum = data["daily"]["relative_humidity_2m_mean"]

# #                         rain_arr = np.nan_to_num(np.array(raw_rain, dtype=np.float32))
# #                         temp_arr = np.nan_to_num(np.array(raw_temp, dtype=np.float32))
# #                         hum_arr = np.nan_to_num(np.array(raw_hum, dtype=np.float32))

# #                         weeks_input = []
# #                         for i in range(27):
# #                             start_idx = i * 7
# #                             end_idx = start_idx + 7
# #                             weeks_input.append([
# #                                 float(np.sum(rain_arr[start_idx:end_idx])),
# #                                 float(np.mean(temp_arr[start_idx:end_idx])),
# #                                 float(np.mean(hum_arr[start_idx:end_idx]))
# #                             ])

# #                                                 # 1. Extract static physical features for the selection
# #                         meta = SPATIAL_REGISTRY[district_choice]
                        
# #                         # Build the 24-dimensional One-Hot row vector component
# #                         one_hot = np.zeros(24, dtype=np.float32)
# #                         one_hot[meta["idx"]] = 1.0

# #                         # Normalize the 6 static features exactly like training bounds
# #                         n_area = (meta["area"] - 4000.0) / (92000.0 - 4000.0)
# #                         n_elev = (meta["elev"] - 85.0) / (1450.0 - 85.0)
# #                         n_ph   = (meta["ph"] - 4.5) / (8.0 - 4.5)
# #                         n_clay = (meta["clay"] - 5.0) / (35.0 - 5.0)
# #                         n_sand = (meta["sand"] - 20.0) / (60.0 - 20.0)
# #                         n_org  = (meta["org"] - 0.5) / (5.0 - 0.5)
# #                         static_30d = np.concatenate([one_hot, [n_area, n_elev, n_ph, n_clay, n_sand, n_org]])

# #                         # 2. Step through weeks, normalize weather, and stitch to reach 33 inputs
# #                         scaled_sequence = []
# #                         for w in weeks_input:
# #                             rain, temp, hum = w[0], w[1], w[2]

# #                             norm_rain = (rain - 0.0) / (365.73 - 0.0)
# #                             norm_temp = (temp - (-14.39)) / (36.926 - (-14.39))
# #                             norm_hum = (hum - 11.41) / (95.437 - 11.41)

# #                             weather_3d = np.array([
# #                                 float(np.clip(norm_rain, 0.0, 1.0)),
# #                                 float(np.clip(norm_temp, 0.0, 1.0)),
# #                                 float(np.clip(norm_hum, 0.0, 1.0))
# #                             ], dtype=np.float32)

# #                             # Concatenate 30 static slots + 3 weather slots = 33 dimensions
# #                             week_33d = np.concatenate([static_30d, weather_3d])
# #                             scaled_sequence.append(week_33d)

# #                         # 3. Create input tensor array of shape [1, 27, 33]
# #                         input_tensor = torch.tensor([scaled_sequence], dtype=torch.float32).to(DEVICE)

# #                         yield_model = load_yield_model()

# #                         if yield_model is None:
# #                             st.error("⚠️ Upgraded model file `spatial_paddy_lstm_final.pth` not found inside your folder directory.")
# #                         else:
# #                             with torch.no_grad():
# #                                 raw_pred = yield_model(input_tensor).cpu().item()

# #                             # Reverse Min/Max training scaling bounds to output real Metric Tons / Hectare
# #                             final_yield = float(raw_pred) * (4.751 - 0.33) + 0.33

# #                             st.write("---")
# #                             st.subheader("📈 Yield Prediction Result")

# #                             r1, r2 = st.columns([1, 1])
# #                             r1.metric(label="Projected Harvest Yield", value=f"{final_yield:.2f} t/ha")
# #                             avg_rain_wk = float(np.mean([w[0] for w in weeks_input]))
# #                             avg_temp_wk = float(np.mean([w[1] for w in weeks_input]))
# #                             r2.metric(label="Avg Weekly Rainfall", value=f"{avg_rain_wk:.1f} mm")

# #                             st.caption(f"Calculated for {district_choice} using updated 33-D physical soil parameters.")

# #                             st.write("**Soil & Site Profile Used In This Prediction**")
# #                             s1, s2, s3 = st.columns(3)
# #                             s1.metric(label="⛰️ Elevation", value=f"{meta['elev']:.0f} m")
# #                             s2.metric(label="🗺️ Cultivated Area", value=f"{meta['area']:,} Ha")
# #                             s3.metric(label="🧪 Topsoil pH", value=f"{meta['ph']:.2f}")

# #                             s4, s5, s6 = st.columns(3)
# #                             s4.metric(label="🌿 Organic Matter", value=f"{meta['org']:.2f} %")
# #                             s5.metric(label="🧱 Clay Content", value=f"{meta['clay']:.1f} %")
# #                             s6.metric(label="⏳ Sand Content", value=f"{meta['sand']:.1f} %")

# #                             st.write("**Soil Texture Breakdown**")
# #                             st.caption(f"Clay (moisture holding): {meta['clay']:.1f}%")
# #                             st.progress(min(max(int(meta['clay'] * 2), 0), 100))
# #                             st.caption(f"Sand (drainage capacity): {meta['sand']:.1f}%")
# #                             st.progress(min(max(int(meta['sand']), 0), 100))

# #                             st.write("**Seasonal Weather Summary (used in prediction)**")
# #                             w1, w2, w3 = st.columns(3)
# #                             w1.metric(label="🌧️ Avg Weekly Rain", value=f"{avg_rain_wk:.1f} mm")
# #                             w2.metric(label="🌡️ Avg Temperature", value=f"{avg_temp_wk:.1f} °C")
# #                             w3.metric(label="📅 Weeks Analyzed", value="27")

# #                             with st.expander("How this yield number was calculated"):
# #                                 st.write(
# #                                     "The model combines a 24-district one-hot location vector, "
# #                                     "6 normalized static soil/site features (area, elevation, pH, clay, "
# #                                     "sand, organic matter), and 3 weekly weather features "
# #                                     "(rainfall, temperature, humidity) across 27 weeks into a "
# #                                     "33-dimensional sequence, then feeds it through the LSTM to "
# #                                     "predict yield in metric tons per hectare."
# #                                 )
# #                             st.write("---")
# #                     else:
# #                         st.error("Could not decode local satellite streaming response arrays. Please try again.")
# #             except Exception as e:
# #                 st.error(f"Network Error: {str(e)}")

# import streamlit as st
# import torch
# import torch.nn as nn
# import timm
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np
# import requests

# # ============================================================
# # CONFIG, MODEL DEFINITIONS & HARDWARE MAPPING
# # ============================================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PLANT_CONFIDENCE_THRESHOLD = 70.0
# GATEKEEPER_PATH = "efficientnet_b0_gatekeeper.pth"
# NUM_PLANT_CLASSES = 14

# # folder-name -> plant species mapping
# PLANT_CLASS_TO_IDX = {
#     'Apple': 0, 'Blueberry': 1, 'Cherry': 2, 'Corn': 3, 'Grape': 4,
#     'Orange': 5, 'Peach': 6, 'Pepper,': 7, 'Potato': 8, 'Raspberry': 9,
#     'Soybean': 10, 'Squash': 11, 'Strawberry': 12, 'Tomato': 13
# }
# IDX_TO_PLANT = {v: k for k, v in PLANT_CLASS_TO_IDX.items()}

# DISEASE_CLASSES = {
#     "Apple": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Blueberry": ['Blueberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Cherry": ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Corn": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Grape": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Orange": ['Orange___Haunglongbing_(Citrus_greening)', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Peach": ['Peach___Bacterial_spot', 'Peach___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Pepper,": ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Raspberry": ['Raspberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Soybean": ['Soybean___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Squash": ['Squash___Powdery_mildew', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Strawberry": ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
#     "Tomato": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
# }

# def disease_model_filename(plant: str) -> str:
#     return f"{plant}_disease_cnn.pth"

# class PlantDiseaseCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(PlantDiseaseCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 28 * 28, 256)
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = self.pool3(self.relu3(self.conv3(x)))
#         x = torch.flatten(x, 1)
#         x = self.relu4(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class DeeperPaddyYieldLSTM(nn.Module):
#     # Change default input_dim constraint from 3 to 33
#     def __init__(self, input_dim=33, hidden_dim=128, num_layers=2, output_dim=1):
#         super(DeeperPaddyYieldLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # Input tensor shape will be: [1, 27, 33]
#         out, _ = self.lstm(x)
#         return self.fc(self.dropout(out[:, -1, :]))


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# @st.cache_resource
# def load_gatekeeper():
#     model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_PLANT_CLASSES)
#     num_features = model.classifier.in_features
#     model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_features, NUM_PLANT_CLASSES))
#     model.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=DEVICE))
#     return model.to(DEVICE).eval()

# @st.cache_resource
# def load_disease_model(plant: str):
#     filename = disease_model_filename(plant)
#     if not os.path.exists(filename):
#         return None, filename
#     model = PlantDiseaseCNN(num_classes=len(DISEASE_CLASSES[plant]))
#     checkpoint = torch.load(filename, map_location=DEVICE)

#     def is_state_dict(d):
#         return isinstance(d, dict) and len(d) > 0 and all(
#             isinstance(v, torch.Tensor) for v in d.values()
#         )

#     # Unwrap nested checkpoint wrappers (any depth, any common key name)
#     # until we reach a dict whose values are all tensors.
#     seen_keys = []
#     while isinstance(checkpoint, dict) and not is_state_dict(checkpoint):
#         seen_keys.append(list(checkpoint.keys()))
#         for key in ("state_dict", "model_state_dict", "model", "weights"):
#             if key in checkpoint:
#                 checkpoint = checkpoint[key]
#                 break
#         else:
#             raise RuntimeError(
#                 f"Could not find a tensor state_dict inside '{filename}'. "
#                 f"Wrapper keys encountered at each level: {seen_keys}. "
#                 f"Inspect the file with torch.load to find the correct key."
#             )

#     model.load_state_dict(checkpoint)
#     return model.to(DEVICE).eval(), filename


# @st.cache_resource
# def load_yield_model():
#     # Update to point to your new Version 2.0 weights file
#     model_path = "spatial_paddy_lstm_final.pth"
#     if not os.path.exists(model_path):
#         return None
    
#     # Initialize the architecture frame with input_dim=33
#     model = DeeperPaddyYieldLSTM(input_dim=33, hidden_dim=128, num_layers=2, output_dim=1)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     return model.to(DEVICE).eval()


# # ============================================================
# # STREAMLIT MAIN VISUAL APP LAYOUT
# # ============================================================
# st.set_page_config(page_title="Agri-Insight Nepal", page_icon="🌾")
# st.title("🌾 Agri-Insight Nepal: Dashboard")
# st.write("Unified deep learning analysis tools for agricultural monitoring in Nepal.")

# tab1, tab2 = st.tabs(["🌿 Leaf Disease Detector", "📊 Paddy Yield Predictor"])

# # ============================================================
# # TAB 1: DISEASE DETECTOR VIEW (Original Code Preserved)
# # ============================================================
# with tab1:
#     st.header("Leaf Health Diagnostics")
#     uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", width='stretch')

#         if st.button("Predict"):
#             img_tensor = transform(image).unsqueeze(0).to(DEVICE)

#             with st.spinner("Identifying plant species..."):
#                 gatekeeper = load_gatekeeper()
#                 with torch.no_grad():
#                     plant_logits = gatekeeper(img_tensor)
#                     plant_probs = torch.softmax(plant_logits, dim=1)[0]
#                     plant_idx = torch.argmax(plant_probs).item()
#                     plant_name = IDX_TO_PLANT[plant_idx]
#                     plant_confidence = plant_probs[plant_idx].item() * 100

#             if plant_confidence < PLANT_CONFIDENCE_THRESHOLD:
#                 st.error(f"⚠️ **Image Rejected** (Plant Recognition Confidence: {plant_confidence:.2f}%). The model is not sure what kind of leaf this is.")
#             else:
#                 st.success(f"**Step 1 — Detected Plant: {plant_name}**  (confidence: {plant_confidence:.2f}%)")

#                 with st.spinner(f"Loading {plant_name} disease model and predicting..."):
#                     disease_model, filename = load_disease_model(plant_name)

#                 if disease_model is None:
#                     st.error(f"Disease model file `{filename}` not found. Place it in the same folder as this app.")
#                 else:
#                     with torch.no_grad():
#                         disease_logits = disease_model(img_tensor)
#                         disease_probs = torch.softmax(disease_logits, dim=1)[0]
#                         disease_idx = torch.argmax(disease_probs).item()
#                         disease_label = DISEASE_CLASSES[plant_name][disease_idx]
#                         disease_confidence = disease_probs[disease_idx].item() * 100

#                     if disease_confidence < PLANT_CONFIDENCE_THRESHOLD:
#                         st.warning(f"⚠️ **Prediction Rejected** (Confidence: {disease_confidence:.2f}%). The Specialist model is too uncertain.")
#                     else:
#                         display_disease = disease_label.split("___", 1)[-1].replace("_", " ")
#                         st.success(f"**Step 2 — Diagnosis: {display_disease}**  (confidence: {disease_confidence:.2f}%)")

#                     top_k = min(3, len(DISEASE_CLASSES[plant_name]))
#                     topk = torch.topk(disease_probs, top_k)
#                     st.write("Top predictions:")
#                     for prob, idx in zip(topk.values, topk.indices):
#                         label = DISEASE_CLASSES[plant_name][idx.item()].split("___", 1)[-1].replace("_", " ")
#                         st.write(f"- {label}: {prob.item()*100:.2f}%")

#     st.divider()
#     with st.expander("Required disease model files"):
#         st.write(f"- `{GATEKEEPER_PATH}` (gatekeeper)")
#         for plant in DISEASE_CLASSES:
#             st.write(f"- `{disease_model_filename(plant)}` ({len(DISEASE_CLASSES[plant])} classes)")

# # ============================================================
# # TAB 2: YIELD PREDICTOR VIEW (Live Weather Mode - Fixed)
# # ============================================================
# with tab2:
#     st.header("🌾 Real-Time Seasonal Yield Estimation")
#     st.write("Fetches live climate data for the selected district to predict expected Paddy yield.")

#     # Expanded district coordinate lookup map covering Nepal's major agricultural zones
#     district_coords = {
#         "Jhapa": {"lat": 26.63, "lon": 87.90},
#         "Morang": {"lat": 26.65, "lon": 87.42},
#         "Sunsari": {"lat": 26.62, "lon": 87.15},
#         "Chitwan": {"lat": 27.58, "lon": 84.49},
#         "Bara": {"lat": 27.05, "lon": 85.02},
#         "Parsa": {"lat": 27.15, "lon": 84.88},
#         "Rautahat": {"lat": 26.84, "lon": 85.26},
#         "Dhanusha": {"lat": 26.78, "lon": 85.97},
#         "Siraha": {"lat": 26.65, "lon": 86.21},
#         "Saptari": {"lat": 26.54, "lon": 86.74},
#         "Rupandehi": {"lat": 27.53, "lon": 83.45},
#         "Kapilvastu": {"lat": 27.53, "lon": 82.95},
#         "Nawalparasi": {"lat": 27.53, "lon": 83.98},
#         "Banke": {"lat": 28.13, "lon": 81.65},
#         "Bardiya": {"lat": 28.32, "lon": 81.36},
#         "Kailali": {"lat": 28.78, "lon": 80.86},
#         "Kanchanpur": {"lat": 28.92, "lon": 80.33},
#         "Dang": {"lat": 28.01, "lon": 82.31},
#         "Kaski": {"lat": 28.25, "lon": 83.98},
#         "Kavre": {"lat": 27.52, "lon": 85.55},
#         "Bhaktapur": {"lat": 27.67, "lon": 85.43},
#         "Lalitpur": {"lat": 27.60, "lon": 85.33},
#         "Kathmandu": {"lat": 27.71, "lon": 85.32},
#         "Palpa": {"lat": 27.86, "lon": 83.55}
#     }

#     # Static Soil and Physical Registry (NARC derived metrics)
#     SPATIAL_REGISTRY = {
#         "Jhapa":       {"idx": 0,  "area": 88500.0, "elev": 125.0, "ph": 6.0282, "clay": 18.7874, "sand": 44.3958, "org": 3.7338},
#         "Bara":        {"idx": 4,  "area": 54680.0, "elev": 95.0,  "ph": 7.2795, "clay": 21.2574, "sand": 35.5916, "org": 1.9240},
#         "Palpa":       {"idx": 23, "area": 9430.0,  "elev": 360.0, "ph": 6.5733, "clay": 13.7497, "sand": 42.7099, "org": 2.9248},
#         "Siraha":      {"idx": 8,  "area": 40000.0, "elev": 105.0, "ph": 6.2000, "clay": 25.0000, "sand": 43.0000, "org": 1.6000},
#         "Parsa":       {"idx": 5,  "area": 45000.0, "elev": 115.0, "ph": 7.1500, "clay": 22.0000, "sand": 36.0000, "org": 1.8000},
#         "Morang":      {"idx": 1,  "area": 81000.0, "elev": 130.0, "ph": 5.6000, "clay": 24.0000, "sand": 42.0000, "org": 2.0000},
#         "Sunsari":     {"idx": 2,  "area": 53000.0, "elev": 140.0, "ph": 5.8000, "clay": 22.0000, "sand": 45.0000, "org": 1.9000},
#         "Chitwan":     {"idx": 3,  "area": 29000.0, "elev": 208.0, "ph": 6.0000, "clay": 20.0000, "sand": 40.0000, "org": 2.4000},
#         "Rautahat":    {"idx": 6,  "area": 39000.0, "elev": 85.0,  "ph": 6.1000, "clay": 27.0000, "sand": 38.0000, "org": 1.6000},
#         "Dhanusha":    {"idx": 7,  "area": 46000.0, "elev": 92.0,  "ph": 6.4000, "clay": 26.0000, "sand": 41.0000, "org": 1.5000},
#         "Saptari":     {"idx": 9,  "area": 41000.0, "elev": 88.0,  "ph": 6.0000, "clay": 26.0000, "sand": 42.0000, "org": 1.7000},
#         "Rupandehi":   {"idx": 10, "area": 68000.0, "elev": 120.0, "ph": 6.8000, "clay": 30.0000, "sand": 30.0000, "org": 2.2000},
#         "Kapilvastu":  {"idx": 11, "area": 71000.0, "elev": 110.0, "ph": 6.7000, "clay": 28.0000, "sand": 34.0000, "org": 2.0000},
#         "Nawalparasi": {"idx": 12, "area": 42000.0, "elev": 150.0, "ph": 6.4000, "clay": 24.0000, "sand": 38.0000, "org": 2.1000},
#         "Banke":       {"idx": 13, "area": 36000.0, "elev": 145.0, "ph": 7.0000, "clay": 22.0000, "sand": 48.0000, "org": 1.8000},
#         "Bardiya":     {"idx": 14, "area": 52000.0, "elev": 155.0, "ph": 6.9000, "clay": 23.0000, "sand": 46.0000, "org": 1.9000},
#         "Kailali":     {"idx": 15, "area": 72000.0, "elev": 190.0, "ph": 6.5000, "clay": 25.0000, "sand": 40.0000, "org": 2.0000},
#         "Kanchanpur":  {"idx": 16, "area": 48000.0, "elev": 210.0, "ph": 6.3000, "clay": 24.0000, "sand": 41.0000, "org": 2.2000},
#         "Dang":        {"idx": 17, "area": 37000.0, "elev": 650.0, "ph": 6.6000, "clay": 21.0000, "sand": 44.0000, "org": 2.3000},
#         "Kaski":       {"idx": 18, "area": 25000.0, "elev": 890.0, "ph": 5.2000, "clay": 18.0000, "sand": 42.0000, "org": 3.2000},
#         "Kavre":       {"idx": 19, "area": 11000.0, "elev": 1450., "ph": 5.1000, "clay": 26.0000, "sand": 34.0000, "org": 2.8000},
#         "Bhaktapur":   {"idx": 20, "area": 4000.0,  "elev": 1330., "ph": 5.4000, "clay": 28.0000, "sand": 30.0000, "org": 2.6000},
#         "Lalitpur":    {"idx": 21, "area": 9000.0,  "elev": 1350., "ph": 5.3000, "clay": 27.0000, "sand": 32.0000, "org": 2.7000},
#         "Kathmandu":   {"idx": 22, "area": 12000.0, "elev": 1340., "ph": 5.5000, "clay": 25.0000, "sand": 35.0000, "org": 2.5000}
#     }

#     district_choice = st.selectbox("Select target Nepal District", list(district_coords.keys()))
#     crop_choice = st.selectbox("Select Target Crop", ["Paddy (Rice)", "Maize (Coming Soon)", "Wheat (Coming Soon)"])

#     if st.button("Calculate Expected Yield"):
#         with st.spinner("Connecting to global satellite feeds to pull real-world seasonal data..."):
#             coords = district_coords[district_choice]
#             base_url = "https://archive-api.open-meteo.com/v1/archive"

#             query_params = {
#                 "latitude": coords["lat"],
#                 "longitude": coords["lon"],
#                 "start_date": "2024-06-01",
#                 "end_date": "2024-11-30",
#                 "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
#                 "timezone": "auto"
#             }

#             try:
#                 headers = {"Accept": "application/json"}
#                 response = requests.get(base_url, params=query_params, headers=headers, timeout=15)

#                 if response.status_code != 200:
#                     st.error(f"Weather Server returned an unexpected error status: Code {response.status_code}. Please try again later.")
#                 else:
#                     data = response.json()

#                     if "daily" in data:
#                         raw_rain = data["daily"]["precipitation_sum"]
#                         raw_temp = data["daily"]["temperature_2m_mean"]
#                         raw_hum = data["daily"]["relative_humidity_2m_mean"]

#                         rain_arr = np.nan_to_num(np.array(raw_rain, dtype=np.float32))
#                         temp_arr = np.nan_to_num(np.array(raw_temp, dtype=np.float32))
#                         hum_arr = np.nan_to_num(np.array(raw_hum, dtype=np.float32))

#                         weeks_input = []
#                         for i in range(27):
#                             start_idx = i * 7
#                             end_idx = start_idx + 7
#                             weeks_input.append([
#                                 float(np.sum(rain_arr[start_idx:end_idx])),
#                                 float(np.mean(temp_arr[start_idx:end_idx])),
#                                 float(np.mean(hum_arr[start_idx:end_idx]))
#                             ])

#                                                 # 1. Extract static physical features for the selection
#                         meta = SPATIAL_REGISTRY[district_choice]
                        
#                         # Build the 24-dimensional One-Hot row vector component
#                         one_hot = np.zeros(24, dtype=np.float32)
#                         one_hot[meta["idx"]] = 1.0

#                         # Normalize the 6 static features exactly like training bounds
#                         n_area = (meta["area"] - 4000.0) / (92000.0 - 4000.0)
#                         n_elev = (meta["elev"] - 85.0) / (1450.0 - 85.0)
#                         n_ph   = (meta["ph"] - 4.5) / (8.0 - 4.5)
#                         n_clay = (meta["clay"] - 5.0) / (35.0 - 5.0)
#                         n_sand = (meta["sand"] - 20.0) / (60.0 - 20.0)
#                         n_org  = (meta["org"] - 0.5) / (5.0 - 0.5)
#                         static_30d = np.concatenate([one_hot, [n_area, n_elev, n_ph, n_clay, n_sand, n_org]])

#                         # 2. Step through weeks, normalize weather, and stitch to reach 33 inputs
#                         scaled_sequence = []
#                         for w in weeks_input:
#                             rain, temp, hum = w[0], w[1], w[2]

#                             norm_rain = (rain - 0.0) / (365.73 - 0.0)
#                             norm_temp = (temp - (-14.39)) / (36.926 - (-14.39))
#                             norm_hum = (hum - 11.41) / (95.437 - 11.41)

#                             weather_3d = np.array([
#                                 float(np.clip(norm_rain, 0.0, 1.0)),
#                                 float(np.clip(norm_temp, 0.0, 1.0)),
#                                 float(np.clip(norm_hum, 0.0, 1.0))
#                             ], dtype=np.float32)

#                             # Concatenate 30 static slots + 3 weather slots = 33 dimensions
#                             week_33d = np.concatenate([static_30d, weather_3d])
#                             scaled_sequence.append(week_33d)

#                         # 3. Create input tensor array of shape [1, 27, 33]
#                         input_tensor = torch.tensor([scaled_sequence], dtype=torch.float32).to(DEVICE)

#                         yield_model = load_yield_model()

#                         if yield_model is None:
#                             st.error("⚠️ Upgraded model file `spatial_paddy_lstm_final.pth` not found inside your folder directory.")
#                         else:
#                             with torch.no_grad():
#                                 raw_pred = yield_model(input_tensor).cpu().item()

#                             # Reverse Min/Max training scaling bounds to output real Metric Tons / Hectare
#                             final_yield = float(raw_pred) * (4.751 - 0.33) + 0.33

#                             st.write("---")
#                             st.subheader("📈 Yield Prediction Result")

#                             r1, r2 = st.columns([1, 1])
#                             r1.metric(label="Projected Harvest Yield", value=f"{final_yield:.2f} t/ha")
#                             avg_rain_wk = float(np.mean([w[0] for w in weeks_input]))
#                             avg_temp_wk = float(np.mean([w[1] for w in weeks_input]))
#                             r2.metric(label="Avg Weekly Rainfall", value=f"{avg_rain_wk:.1f} mm")

#                             st.caption(f"Calculated for {district_choice} using updated 33-D physical soil parameters.")

#                             st.write("**Soil & Site Profile Used In This Prediction**")
#                             s1, s2, s3 = st.columns(3)
#                             s1.metric(label="⛰️ Elevation", value=f"{meta['elev']:.0f} m")
#                             s2.metric(label="🗺️ Cultivated Area", value=f"{meta['area']:,} Ha")
#                             s3.metric(label="🧪 Topsoil pH", value=f"{meta['ph']:.2f}")

#                             s4, s5, s6 = st.columns(3)
#                             s4.metric(label="🌿 Organic Matter", value=f"{meta['org']:.2f} %")
#                             s5.metric(label="🧱 Clay Content", value=f"{meta['clay']:.1f} %")
#                             s6.metric(label="⏳ Sand Content", value=f"{meta['sand']:.1f} %")

#                             st.write("**Soil Texture Breakdown**")
#                             st.caption(f"Clay (moisture holding): {meta['clay']:.1f}%")
#                             st.progress(min(max(int(meta['clay'] * 2), 0), 100))
#                             st.caption(f"Sand (drainage capacity): {meta['sand']:.1f}%")
#                             st.progress(min(max(int(meta['sand']), 0), 100))

#                             st.write("**Seasonal Weather Summary (used in prediction)**")
#                             w1, w2, w3 = st.columns(3)
#                             w1.metric(label="🌧️ Avg Weekly Rain", value=f"{avg_rain_wk:.1f} mm")
#                             w2.metric(label="🌡️ Avg Temperature", value=f"{avg_temp_wk:.1f} °C")
#                             w3.metric(label="📅 Weeks Analyzed", value="27")

#                             with st.expander("How this yield number was calculated"):
#                                 st.write(
#                                     "The model combines a 24-district one-hot location vector, "
#                                     "6 normalized static soil/site features (area, elevation, pH, clay, "
#                                     "sand, organic matter), and 3 weekly weather features "
#                                     "(rainfall, temperature, humidity) across 27 weeks into a "
#                                     "33-dimensional sequence, then feeds it through the LSTM to "
#                                     "predict yield in metric tons per hectare."
#                                 )
#                             st.write("---")
#                     else:
#                         st.error("Could not decode local satellite streaming response arrays. Please try again.")
#             except Exception as e:
#                 st.error(f"Network Error: {str(e)}")

import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import requests

# ============================================================
# CONFIG, MODEL DEFINITIONS & HARDWARE MAPPING
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANT_CONFIDENCE_THRESHOLD = 70.0
GATEKEEPER_PATH = "efficientnet_b0_gatekeeper.pth"
NUM_PLANT_CLASSES = 14

# folder-name -> plant species mapping
PLANT_CLASS_TO_IDX = {
    'Apple': 0, 'Blueberry': 1, 'Cherry': 2, 'Corn': 3, 'Grape': 4,
    'Orange': 5, 'Peach': 6, 'Pepper,': 7, 'Potato': 8, 'Raspberry': 9,
    'Soybean': 10, 'Squash': 11, 'Strawberry': 12, 'Tomato': 13
}
IDX_TO_PLANT = {v: k for k, v in PLANT_CLASS_TO_IDX.items()}

DISEASE_CLASSES = {
    "Apple": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Blueberry": ['Blueberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Cherry": ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Corn": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Grape": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Orange": ['Orange___Haunglongbing_(Citrus_greening)', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Peach": ['Peach___Bacterial_spot', 'Peach___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Pepper,": ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Raspberry": ['Raspberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Soybean": ['Soybean___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Squash": ['Squash___Powdery_mildew', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Strawberry": ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid', 'Invalid'],
    "Tomato": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
}

def disease_model_filename(plant: str) -> str:
    return f"{plant}_disease_cnn.pth"

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

class DeeperPaddyYieldLSTM(nn.Module):
    # Change default input_dim constraint from 3 to 33
    def __init__(self, input_dim=33, hidden_dim=128, num_layers=2, output_dim=1):
        super(DeeperPaddyYieldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input tensor shape will be: [1, 27, 33]
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_gatekeeper():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_PLANT_CLASSES)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_features, NUM_PLANT_CLASSES))
    model.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

@st.cache_resource
def load_disease_model(plant: str):
    filename = disease_model_filename(plant)
    if not os.path.exists(filename):
        return None, filename
    model = PlantDiseaseCNN(num_classes=len(DISEASE_CLASSES[plant]))
    checkpoint = torch.load(filename, map_location=DEVICE)
    expected_keys = set(model.state_dict().keys())

    def is_state_dict(d):
        # A dict whose keys match the model's parameter names (values may
        # be tensors, numpy arrays, or other tensor-like objects).
        return isinstance(d, dict) and len(d) > 0 and set(d.keys()) == expected_keys

    # Unwrap nested checkpoint wrappers (any depth, any common key name)
    # until we reach a dict keyed by the model's parameter names.
    seen_keys = []
    while isinstance(checkpoint, dict) and not is_state_dict(checkpoint):
        seen_keys.append(list(checkpoint.keys()))
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
        else:
            raise RuntimeError(
                f"Could not find a state_dict matching {sorted(expected_keys)} "
                f"inside '{filename}'. Wrapper keys encountered at each level: "
                f"{seen_keys}. Inspect the file with torch.load to find the correct key."
            )

    # Coerce every value to a real torch.Tensor (covers numpy arrays,
    # lists, or tensors saved on a different dtype/device) before loading.
    checkpoint = {k: torch.as_tensor(v).to(DEVICE) for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model.to(DEVICE).eval(), filename


@st.cache_resource
def load_yield_model():
    # Update to point to your new Version 2.0 weights file
    model_path = "spatial_paddy_lstm_final.pth"
    if not os.path.exists(model_path):
        return None
    
    # Initialize the architecture frame with input_dim=33
    model = DeeperPaddyYieldLSTM(input_dim=33, hidden_dim=128, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ============================================================
# STREAMLIT MAIN VISUAL APP LAYOUT
# ============================================================
st.set_page_config(page_title="Agri-Insight Nepal", page_icon="🌾")
st.title("🌾 Agri-Insight Nepal: Dashboard")
st.write("Unified deep learning analysis tools for agricultural monitoring in Nepal.")

tab1, tab2 = st.tabs(["🌿 Leaf Disease Detector", "📊 Paddy Yield Predictor"])

# ============================================================
# TAB 1: DISEASE DETECTOR VIEW (Original Code Preserved)
# ============================================================
with tab1:
    st.header("Leaf Health Diagnostics")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width='stretch')

        if st.button("Predict"):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with st.spinner("Identifying plant species..."):
                gatekeeper = load_gatekeeper()
                with torch.no_grad():
                    plant_logits = gatekeeper(img_tensor)
                    plant_probs = torch.softmax(plant_logits, dim=1)[0]
                    plant_idx = torch.argmax(plant_probs).item()
                    plant_name = IDX_TO_PLANT[plant_idx]
                    plant_confidence = plant_probs[plant_idx].item() * 100

            if plant_confidence < PLANT_CONFIDENCE_THRESHOLD:
                st.error(f"⚠️ **Image Rejected** (Plant Recognition Confidence: {plant_confidence:.2f}%). The model is not sure what kind of leaf this is.")
            else:
                st.success(f"**Step 1 — Detected Plant: {plant_name}**  (confidence: {plant_confidence:.2f}%)")

                with st.spinner(f"Loading {plant_name} disease model and predicting..."):
                    disease_model, filename = load_disease_model(plant_name)

                if disease_model is None:
                    st.error(f"Disease model file `{filename}` not found. Place it in the same folder as this app.")
                else:
                    with torch.no_grad():
                        disease_logits = disease_model(img_tensor)
                        disease_probs = torch.softmax(disease_logits, dim=1)[0]
                        disease_idx = torch.argmax(disease_probs).item()
                        disease_label = DISEASE_CLASSES[plant_name][disease_idx]
                        disease_confidence = disease_probs[disease_idx].item() * 100

                    if disease_confidence < PLANT_CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ **Prediction Rejected** (Confidence: {disease_confidence:.2f}%). The Specialist model is too uncertain.")
                    else:
                        display_disease = disease_label.split("___", 1)[-1].replace("_", " ")
                        st.success(f"**Step 2 — Diagnosis: {display_disease}**  (confidence: {disease_confidence:.2f}%)")

                    top_k = min(3, len(DISEASE_CLASSES[plant_name]))
                    topk = torch.topk(disease_probs, top_k)
                    st.write("Top predictions:")
                    for prob, idx in zip(topk.values, topk.indices):
                        label = DISEASE_CLASSES[plant_name][idx.item()].split("___", 1)[-1].replace("_", " ")
                        st.write(f"- {label}: {prob.item()*100:.2f}%")

    st.divider()
    with st.expander("Required disease model files"):
        st.write(f"- `{GATEKEEPER_PATH}` (gatekeeper)")
        for plant in DISEASE_CLASSES:
            st.write(f"- `{disease_model_filename(plant)}` ({len(DISEASE_CLASSES[plant])} classes)")

# ============================================================
# TAB 2: YIELD PREDICTOR VIEW (Live Weather Mode - Fixed)
# ============================================================
with tab2:
    st.header("🌾 Real-Time Seasonal Yield Estimation")
    st.write("Fetches live climate data for the selected district to predict expected Paddy yield.")

    # Expanded district coordinate lookup map covering Nepal's major agricultural zones
    district_coords = {
        "Jhapa": {"lat": 26.63, "lon": 87.90},
        "Morang": {"lat": 26.65, "lon": 87.42},
        "Sunsari": {"lat": 26.62, "lon": 87.15},
        "Chitwan": {"lat": 27.58, "lon": 84.49},
        "Bara": {"lat": 27.05, "lon": 85.02},
        "Parsa": {"lat": 27.15, "lon": 84.88},
        "Rautahat": {"lat": 26.84, "lon": 85.26},
        "Dhanusha": {"lat": 26.78, "lon": 85.97},
        "Siraha": {"lat": 26.65, "lon": 86.21},
        "Saptari": {"lat": 26.54, "lon": 86.74},
        "Rupandehi": {"lat": 27.53, "lon": 83.45},
        "Kapilvastu": {"lat": 27.53, "lon": 82.95},
        "Nawalparasi": {"lat": 27.53, "lon": 83.98},
        "Banke": {"lat": 28.13, "lon": 81.65},
        "Bardiya": {"lat": 28.32, "lon": 81.36},
        "Kailali": {"lat": 28.78, "lon": 80.86},
        "Kanchanpur": {"lat": 28.92, "lon": 80.33},
        "Dang": {"lat": 28.01, "lon": 82.31},
        "Kaski": {"lat": 28.25, "lon": 83.98},
        "Kavre": {"lat": 27.52, "lon": 85.55},
        "Bhaktapur": {"lat": 27.67, "lon": 85.43},
        "Lalitpur": {"lat": 27.60, "lon": 85.33},
        "Kathmandu": {"lat": 27.71, "lon": 85.32},
        "Palpa": {"lat": 27.86, "lon": 83.55}
    }

    # Static Soil and Physical Registry (NARC derived metrics)
    SPATIAL_REGISTRY = {
        "Jhapa":       {"idx": 0,  "area": 88500.0, "elev": 125.0, "ph": 6.0282, "clay": 18.7874, "sand": 44.3958, "org": 3.7338},
        "Bara":        {"idx": 4,  "area": 54680.0, "elev": 95.0,  "ph": 7.2795, "clay": 21.2574, "sand": 35.5916, "org": 1.9240},
        "Palpa":       {"idx": 23, "area": 9430.0,  "elev": 360.0, "ph": 6.5733, "clay": 13.7497, "sand": 42.7099, "org": 2.9248},
        "Siraha":      {"idx": 8,  "area": 40000.0, "elev": 105.0, "ph": 6.2000, "clay": 25.0000, "sand": 43.0000, "org": 1.6000},
        "Parsa":       {"idx": 5,  "area": 45000.0, "elev": 115.0, "ph": 7.1500, "clay": 22.0000, "sand": 36.0000, "org": 1.8000},
        "Morang":      {"idx": 1,  "area": 81000.0, "elev": 130.0, "ph": 5.6000, "clay": 24.0000, "sand": 42.0000, "org": 2.0000},
        "Sunsari":     {"idx": 2,  "area": 53000.0, "elev": 140.0, "ph": 5.8000, "clay": 22.0000, "sand": 45.0000, "org": 1.9000},
        "Chitwan":     {"idx": 3,  "area": 29000.0, "elev": 208.0, "ph": 6.0000, "clay": 20.0000, "sand": 40.0000, "org": 2.4000},
        "Rautahat":    {"idx": 6,  "area": 39000.0, "elev": 85.0,  "ph": 6.1000, "clay": 27.0000, "sand": 38.0000, "org": 1.6000},
        "Dhanusha":    {"idx": 7,  "area": 46000.0, "elev": 92.0,  "ph": 6.4000, "clay": 26.0000, "sand": 41.0000, "org": 1.5000},
        "Saptari":     {"idx": 9,  "area": 41000.0, "elev": 88.0,  "ph": 6.0000, "clay": 26.0000, "sand": 42.0000, "org": 1.7000},
        "Rupandehi":   {"idx": 10, "area": 68000.0, "elev": 120.0, "ph": 6.8000, "clay": 30.0000, "sand": 30.0000, "org": 2.2000},
        "Kapilvastu":  {"idx": 11, "area": 71000.0, "elev": 110.0, "ph": 6.7000, "clay": 28.0000, "sand": 34.0000, "org": 2.0000},
        "Nawalparasi": {"idx": 12, "area": 42000.0, "elev": 150.0, "ph": 6.4000, "clay": 24.0000, "sand": 38.0000, "org": 2.1000},
        "Banke":       {"idx": 13, "area": 36000.0, "elev": 145.0, "ph": 7.0000, "clay": 22.0000, "sand": 48.0000, "org": 1.8000},
        "Bardiya":     {"idx": 14, "area": 52000.0, "elev": 155.0, "ph": 6.9000, "clay": 23.0000, "sand": 46.0000, "org": 1.9000},
        "Kailali":     {"idx": 15, "area": 72000.0, "elev": 190.0, "ph": 6.5000, "clay": 25.0000, "sand": 40.0000, "org": 2.0000},
        "Kanchanpur":  {"idx": 16, "area": 48000.0, "elev": 210.0, "ph": 6.3000, "clay": 24.0000, "sand": 41.0000, "org": 2.2000},
        "Dang":        {"idx": 17, "area": 37000.0, "elev": 650.0, "ph": 6.6000, "clay": 21.0000, "sand": 44.0000, "org": 2.3000},
        "Kaski":       {"idx": 18, "area": 25000.0, "elev": 890.0, "ph": 5.2000, "clay": 18.0000, "sand": 42.0000, "org": 3.2000},
        "Kavre":       {"idx": 19, "area": 11000.0, "elev": 1450., "ph": 5.1000, "clay": 26.0000, "sand": 34.0000, "org": 2.8000},
        "Bhaktapur":   {"idx": 20, "area": 4000.0,  "elev": 1330., "ph": 5.4000, "clay": 28.0000, "sand": 30.0000, "org": 2.6000},
        "Lalitpur":    {"idx": 21, "area": 9000.0,  "elev": 1350., "ph": 5.3000, "clay": 27.0000, "sand": 32.0000, "org": 2.7000},
        "Kathmandu":   {"idx": 22, "area": 12000.0, "elev": 1340., "ph": 5.5000, "clay": 25.0000, "sand": 35.0000, "org": 2.5000}
    }

    district_choice = st.selectbox("Select target Nepal District", list(district_coords.keys()))
    crop_choice = st.selectbox("Select Target Crop", ["Paddy (Rice)", "Maize (Coming Soon)", "Wheat (Coming Soon)"])

    if st.button("Calculate Expected Yield"):
        with st.spinner("Connecting to global satellite feeds to pull real-world seasonal data..."):
            coords = district_coords[district_choice]
            base_url = "https://archive-api.open-meteo.com/v1/archive"

            query_params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": "2024-06-01",
                "end_date": "2024-11-30",
                "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
                "timezone": "auto"
            }

            try:
                headers = {"Accept": "application/json"}
                response = requests.get(base_url, params=query_params, headers=headers, timeout=15)

                if response.status_code != 200:
                    st.error(f"Weather Server returned an unexpected error status: Code {response.status_code}. Please try again later.")
                else:
                    data = response.json()

                    if "daily" in data:
                        raw_rain = data["daily"]["precipitation_sum"]
                        raw_temp = data["daily"]["temperature_2m_mean"]
                        raw_hum = data["daily"]["relative_humidity_2m_mean"]

                        rain_arr = np.nan_to_num(np.array(raw_rain, dtype=np.float32))
                        temp_arr = np.nan_to_num(np.array(raw_temp, dtype=np.float32))
                        hum_arr = np.nan_to_num(np.array(raw_hum, dtype=np.float32))

                        weeks_input = []
                        for i in range(27):
                            start_idx = i * 7
                            end_idx = start_idx + 7
                            weeks_input.append([
                                float(np.sum(rain_arr[start_idx:end_idx])),
                                float(np.mean(temp_arr[start_idx:end_idx])),
                                float(np.mean(hum_arr[start_idx:end_idx]))
                            ])

                                                # 1. Extract static physical features for the selection
                        meta = SPATIAL_REGISTRY[district_choice]
                        
                        # Build the 24-dimensional One-Hot row vector component
                        one_hot = np.zeros(24, dtype=np.float32)
                        one_hot[meta["idx"]] = 1.0

                        # Normalize the 6 static features exactly like training bounds
                        n_area = (meta["area"] - 4000.0) / (92000.0 - 4000.0)
                        n_elev = (meta["elev"] - 85.0) / (1450.0 - 85.0)
                        n_ph   = (meta["ph"] - 4.5) / (8.0 - 4.5)
                        n_clay = (meta["clay"] - 5.0) / (35.0 - 5.0)
                        n_sand = (meta["sand"] - 20.0) / (60.0 - 20.0)
                        n_org  = (meta["org"] - 0.5) / (5.0 - 0.5)
                        static_30d = np.concatenate([one_hot, [n_area, n_elev, n_ph, n_clay, n_sand, n_org]])

                        # 2. Step through weeks, normalize weather, and stitch to reach 33 inputs
                        scaled_sequence = []
                        for w in weeks_input:
                            rain, temp, hum = w[0], w[1], w[2]

                            norm_rain = (rain - 0.0) / (365.73 - 0.0)
                            norm_temp = (temp - (-14.39)) / (36.926 - (-14.39))
                            norm_hum = (hum - 11.41) / (95.437 - 11.41)

                            weather_3d = np.array([
                                float(np.clip(norm_rain, 0.0, 1.0)),
                                float(np.clip(norm_temp, 0.0, 1.0)),
                                float(np.clip(norm_hum, 0.0, 1.0))
                            ], dtype=np.float32)

                            # Concatenate 30 static slots + 3 weather slots = 33 dimensions
                            week_33d = np.concatenate([static_30d, weather_3d])
                            scaled_sequence.append(week_33d)

                        # 3. Create input tensor array of shape [1, 27, 33]
                        input_tensor = torch.tensor([scaled_sequence], dtype=torch.float32).to(DEVICE)

                        yield_model = load_yield_model()

                        if yield_model is None:
                            st.error("⚠️ Upgraded model file `spatial_paddy_lstm_final.pth` not found inside your folder directory.")
                        else:
                            with torch.no_grad():
                                raw_pred = yield_model(input_tensor).cpu().item()

                            # Reverse Min/Max training scaling bounds to output real Metric Tons / Hectare
                            final_yield = float(raw_pred) * (4.751 - 0.33) + 0.33

                            st.write("---")
                            st.subheader("📈 Yield Prediction Result")

                            r1, r2 = st.columns([1, 1])
                            r1.metric(label="Projected Harvest Yield", value=f"{final_yield:.2f} t/ha")
                            avg_rain_wk = float(np.mean([w[0] for w in weeks_input]))
                            avg_temp_wk = float(np.mean([w[1] for w in weeks_input]))
                            r2.metric(label="Avg Weekly Rainfall", value=f"{avg_rain_wk:.1f} mm")

                            st.caption(f"Calculated for {district_choice} using updated 33-D physical soil parameters.")

                            st.write("**Soil & Site Profile Used In This Prediction**")
                            s1, s2, s3 = st.columns(3)
                            s1.metric(label="⛰️ Elevation", value=f"{meta['elev']:.0f} m")
                            s2.metric(label="🗺️ Cultivated Area", value=f"{meta['area']:,} Ha")
                            s3.metric(label="🧪 Topsoil pH", value=f"{meta['ph']:.2f}")

                            s4, s5, s6 = st.columns(3)
                            s4.metric(label="🌿 Organic Matter", value=f"{meta['org']:.2f} %")
                            s5.metric(label="🧱 Clay Content", value=f"{meta['clay']:.1f} %")
                            s6.metric(label="⏳ Sand Content", value=f"{meta['sand']:.1f} %")

                            st.write("**Soil Texture Breakdown**")
                            st.caption(f"Clay (moisture holding): {meta['clay']:.1f}%")
                            st.progress(min(max(int(meta['clay'] * 2), 0), 100))
                            st.caption(f"Sand (drainage capacity): {meta['sand']:.1f}%")
                            st.progress(min(max(int(meta['sand']), 0), 100))

                            st.write("**Seasonal Weather Summary (used in prediction)**")
                            w1, w2, w3 = st.columns(3)
                            w1.metric(label="🌧️ Avg Weekly Rain", value=f"{avg_rain_wk:.1f} mm")
                            w2.metric(label="🌡️ Avg Temperature", value=f"{avg_temp_wk:.1f} °C")
                            w3.metric(label="📅 Weeks Analyzed", value="27")

                            with st.expander("How this yield number was calculated"):
                                st.write(
                                    "The model combines a 24-district one-hot location vector, "
                                    "6 normalized static soil/site features (area, elevation, pH, clay, "
                                    "sand, organic matter), and 3 weekly weather features "
                                    "(rainfall, temperature, humidity) across 27 weeks into a "
                                    "33-dimensional sequence, then feeds it through the LSTM to "
                                    "predict yield in metric tons per hectare."
                                )
                            st.write("---")
                    else:
                        st.error("Could not decode local satellite streaming response arrays. Please try again.")
            except Exception as e:
                st.error(f"Network Error: {str(e)}")