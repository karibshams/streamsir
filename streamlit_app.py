# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- 1. Custom CNN Definition ---

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224→112

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112→56

            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56→28
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


# --- 2. Model Loading Helpers ---

def load_custom_cnn(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load CustomCNN weights and return in eval mode."""
    model = CustomCNN(num_classes).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_resnet50(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load a ResNet-50 with custom final layer."""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


# --- 3. Streamlit App Setup ---

st.title("Pumpkin Leaf Disease Classifier")

# Sidebar: model selection & file uploader
model_choice = st.sidebar.selectbox("Choose model", ("Custom CNN", "ResNet-50"))
uploaded_file = st.sidebar.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# Class names and device
CLASS_NAMES = ["Bacterial Spot", "Downy Mildew", "Healthy", "Mosaic", "Powdery Mildew"]
NUM_CLASSES  = len(CLASS_NAMES)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache the model-loading (long-lived resources)
@st.cache_resource
def load_models():
    cnn   = load_custom_cnn("custom_cnn_model.pth", NUM_CLASSES, DEVICE)
    res50 = load_resnet50("transfer_learning_resnet50.pth", NUM_CLASSES, DEVICE)
    return cnn, res50

custom_cnn, resnet50 = load_models()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])


# --- 4. Prediction & Display ---

if uploaded_file:
    # Load and show the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    # Preprocess
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Choose model & predict
    model = custom_cnn if model_choice == "Custom CNN" else resnet50
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Display top-3 predictions
    top_idxs = np.argsort(probs)[::-1][:3]
    st.subheader("Predictions")
    for idx in top_idxs:
        st.write(f"{CLASS_NAMES[idx]}: {probs[idx]*100:.2f}%")
