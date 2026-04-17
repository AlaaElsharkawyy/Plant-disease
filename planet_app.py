import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# -------------------------
# 🎨 CUSTOM UI + CSS
# -------------------------
st.markdown("""
<style>

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #d4edda;
    border: 2px dashed #28a745;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

/* Upload Button */
[data-testid="stFileUploader"] button {
    background-color: #28a745;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: bold;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #218838;
}

/* Image animation */
img {
    border-radius: 12px;
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}

/* Prediction Card */
.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-top: 20px;
    animation: fadeIn 0.8s ease-in-out;
}

.result-title {
    font-size: 20px;
    font-weight: bold;
    color: #d6336c;
}

.result-text {
    font-size: 16px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Model Definition
# -------------------------
class PlantModel(nn.Module):
    def __init__(self, num_plants, num_diseases):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Linear(512, 256)

        self.plant_head = nn.Linear(256, num_plants)
        self.disease_head = nn.Linear(256, num_diseases)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        plant_out = self.plant_head(x)
        disease_out = self.disease_head(x)

        return plant_out, disease_out


# -------------------------
# Load Model
# -------------------------
model = PlantModel(num_plants=14, num_diseases=21)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Labels
# -------------------------
idx_to_plant = {
    0:'Apple', 1:'Blueberry', 2:'Cherry_(including_sour)', 3:'Corn_(maize)',
    4:'Grape', 5:'Orange', 6:'Peach', 7:'Pepper,_bell', 8:'Potato',
    9:'Raspberry', 10:'Soybean', 11:'Squash', 12:'Strawberry', 13:'Tomato'
}

idx_to_disease = {
    0:'Apple_scab', 1:'Bacterial_spot', 2:'Black_rot', 3:'Cedar_apple_rust',
    4:'Cercospora_leaf_spot Gray_leaf_spot', 5:'Common_rust',
    6:'Early_blight', 7:'Esca_(Black_Measles)',
    8:'Haunglongbing_(Citrus_greening)', 9:'Late_blight',
    10:'Leaf_Mold', 11:'Leaf_blight_(Isariopsis_Leaf_Spot)',
    12:'Leaf_scorch', 13:'Northern_Leaf_Blight',
    14:'Powdery_mildew', 15:'Septoria_leaf_spot',
    16:'Spider_mites Two-spotted_spider_mite',
    17:'Target_Spot', 18:'Tomato_Yellow_Leaf_Curl_Virus',
    19:'Tomato_mosaic_virus', 20:'healthy'
}

# -------------------------
# UI
# -------------------------
st.title("🌿 Plant Disease Detection")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------
# Prediction
# -------------------------
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")
    #st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img).unsqueeze(0)

    with st.spinner("🔍 Analyzing..."):
        with torch.no_grad():
            plant_out, disease_out = model(img_t)

            plant_prob = F.softmax(plant_out, dim=1)
            disease_prob = F.softmax(disease_out, dim=1)

            plant_idx = plant_prob.argmax().item()
            disease_idx = disease_prob.argmax().item()

    # 🎉 Result Card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">🌱 Prediction Result</div>
        <div class="result-text">
            <b>Plant:</b> {idx_to_plant[plant_idx]} ({plant_prob.max().item()*100:.2f}%)
        </div>
        <div class="result-text">
            <b>Disease:</b> {idx_to_disease[disease_idx]} ({disease_prob.max().item()*100:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)