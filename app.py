import streamlit as st
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

from lib.models.resnet50 import ResNet50Classifier
from lib.defaults import Args

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

args = Args
device = torch.device(args.cuda) if torch.cuda.is_available() else torch.device("cpu")

def predict(model, image):
    # Apply transformations to the image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)


    _, predicted = torch.max(outputs.data, 1)
    print(torch.softmax(outputs.data, dim=1))

    if predicted.item() == 1:
        return "Cat"
    else:
        return "Dog"

    
if __name__ == "__main__":
    model = ResNet50Classifier().to(device)
    model.load_state_dict(torch.load("./weights/last.pt", weights_only=True, map_location=torch.device('cpu')))

    # Streamlit UI
    st.title("Cat or Dog Image Classification")
    st.write("Upload an image of a cat or a dog, and the model will predict the label.")

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Get the prediction
        prediction = predict(model, image)
        st.write(f"Prediction: {prediction}")