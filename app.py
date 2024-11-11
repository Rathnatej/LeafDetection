import streamlit as st
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms

# Define the custom model class (if required)
class CombinedModel(nn.Module):
    def __init__(self, vit, resnet):
        super(CombinedModel, self).__init__()
        self.vit = vit
        self.resnet = resnet
        # Fully connected layer after concatenating features
        self.fc = nn.Sequential(
            nn.Linear(768 + 2048, 512),  # Concatenate ViT and ResNet features
            nn.ReLU(),
            nn.Linear(512, len(class_names))  # Output layer for classification
        )

    def forward(self, x):
        # Extract features from ViT and ResNet
        vit_features = self.vit(x)
        resnet_features = self.resnet(x)
        
        # Concatenate the features
        combined_features = torch.cat((vit_features, resnet_features), dim=1)
        
        # Pass through the feedforward network
        output = self.fc(combined_features)
        return output

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('combined_model.pth', map_location=device)
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Define the classes (update according to your dataset)
class_names = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Tungro']

# Streamlit app interface
st.title("Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf, and the model will classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Add a submit button
    if st.button("Submit"):
        st.write("Classifying...")

        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Get prediction from the model
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        # Show the prediction
        st.write(f"The model predicts that the leaf has **{predicted_class}**.")
