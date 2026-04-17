import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# =========================
# تحميل الموديل
# =========================
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# أسماء الكلاسات (عدليها حسب الداتا بتاعتك)
classes = [
    "Apple___Scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy"
]

# =========================
# preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# prediction function
# =========================
def predict(image: Image.Image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    return classes[pred]

# =========================
# UI
# =========================
st.title("🌿 Plant Disease Classifier")
st.write("Upload an image and get prediction from your trained model")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Predicting...")

    label = predict(image)

    st.success(f"Prediction: {label}")