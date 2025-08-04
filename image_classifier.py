import os
import requests
import fitz
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "image_classifier_model.pth"
MEDICAL_DATA_URL = "https://www.medtronic.com/content/dam/medtronic-com/en-us/corporate/images/medical-device-image.jpg"
NON_MEDICAL_DATA_URL = "https://images.unsplash.com/photo-1501785888041-af3ef285b760"

# --- DATASET AND PRE-PROCESSING ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(["medical", "non-medical"]):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- MODEL DEFINITION ---
def get_model():
    # Use a pre-trained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer with a new one
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    return model

# --- TRAINING FUNCTION ---
def train_model(train_dir, val_dir):
    print("Starting model training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, loss, and optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # Only train the new head

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

# --- INFERENCE FUNCTIONS ---
# def get_images_from_url(url):
#     print(f"Extracting images from URL: {url}")
#     try:
#         response = requests.get(url)
#         response.raise_for_status() # Raise an exception for bad status codes
        
#         images = []
        # A simple heuristic to find images in HTML.
        # This part could be more complex using libraries like BeautifulSoup.
        # For this example, we'll try to get the image directly from the URL.
        # You can extend this with a web-scraping logic.
        # if response.headers.get('content-type').startswith('image'):
        #     img = Image.open(io.BytesIO(response.content))
        #     images.append(img)
        
        # A more robust approach for finding images in HTML:
        # from bs4 import BeautifulSoup
        # soup = BeautifulSoup(response.text, 'html.parser')
        # for img_tag in soup.find_all('img'):
        #     img_url = img_tag.get('src')
        #     # Handle relative paths, etc.
        #     if img_url and img_url.startswith('http'):
        #         images.extend(get_images_from_url(img_url))

    #     return images
    # except requests.exceptions.RequestException as e:
    #     print(f"Error fetching URL: {e}")
    #     return []
# Updated get_images_from_url function
def get_images_from_url(url):
    print(f"Extracting images from URL: {url}")
    # Add a User-Agent header to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if 'image' in response.headers.get('content-type', ''):
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            return [img]
        else:
            print("The URL does not appear to be a direct image link.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

def get_images_from_pdf(pdf_path):
    print(f"Extracting images from PDF: {pdf_path}")
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                images.append(img)
        doc.close()
        return images
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        return []

def classify_images(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded trained model.")
    else:
        print("Model not found. Please train the model first.")
        return []

    model.eval()
    class_labels = ["medical", "non-medical"]
    predictions = []

    with torch.no_grad():
        for i, img in enumerate(images):
            start_time = time.time()
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)
            prediction = class_labels[predicted_class.item()]
            end_time = time.time()
            inference_time = end_time - start_time
            predictions.append({
                "image_index": i,
                "prediction": prediction,
                "inference_time": inference_time
            })
    return predictions

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Medical vs. Non-Medical Image Classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--url", type=str, help="URL to classify images from")
    parser.add_argument("--pdf", type=str, help="PDF file path to classify images from")
    
    args = parser.parse_args()

    if args.train:
        # Create dummy data directories for demonstration.
        # For your actual submission, you should replace this with a real dataset.
        os.makedirs("train/medical", exist_ok=True)
        os.makedirs("train/non-medical", exist_ok=True)
        os.makedirs("val/medical", exist_ok=True)
        os.makedirs("val/non-medical", exist_ok=True)
        print("Dummy data folders created. Please add your images to these folders for training.")
        
        train_dir = "train"
        val_dir = "val"
        train_model(train_dir, val_dir)

    elif args.url:
        images = get_images_from_url(args.url)
        if images:
            predictions = classify_images(images)
            print("\n--- CLASSIFICATION RESULTS (URL) ---")
            for pred in predictions:
                print(f"Image {pred['image_index']}: {pred['prediction']} (Inference time: {pred['inference_time']:.4f}s)")
        else:
            print("No images found or could not retrieve content from URL.")
    
    elif args.pdf:
        images = get_images_from_pdf(args.pdf)
        if images:
            predictions = classify_images(images)
            print("\n--- CLASSIFICATION RESULTS (PDF) ---")
            for pred in predictions:
                print(f"Image {pred['image_index']}: {pred['prediction']} (Inference time: {pred['inference_time']:.4f}s)")
        else:
            print("No images found or could not retrieve content from PDF.")

    else:
        print("Please provide a valid argument: --train, --url <URL>, or --pdf <PDF_PATH>")
        print("Example: python image_classifier.py --train")
        print("Example: python image_classifier.py --url https://www.public-medical-website.com/image.jpg")
        print("Example: python image_classifier.py --pdf path/to/your/document.pdf")