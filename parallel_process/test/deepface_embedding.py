import os
import requests
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch import nn

def download_file(url, dest_path):
    """
    Downloads a file from the specified URL to the destination path with a progress bar.

    Args:
        url (str): URL of the file to download.
        dest_path (str): Path where the downloaded file will be saved.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if response.status_code == 200:
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        print(f"\nDownloaded {os.path.basename(dest_path)} successfully.")
    else:
        print(f"Failed to download from {url}. Status code: {response.status_code}")

def download_facenet_model():
    """
    Downloads the FaceNet model weights and stores them locally.
    """
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"  # Example URL
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    dest_path = os.path.join(models_dir, "facenet512_weights.pth")

    if not os.path.exists(dest_path):
        print(f"Starting download from {url}...")
        download_file(url, dest_path)
    else:
        print(f"The file {dest_path} already exists. Skipping download.")

# Define the InceptionResnetV1 model class based on DeepFace's implementation
class InceptionResnetV1(nn.Module):
    def __init__(self, pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6):
        super(InceptionResnetV1, self).__init__()
        # Load the model architecture and weights
        # This is a simplified version; the actual implementation may vary
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)
        self.dropout = nn.Dropout(dropout_prob)
        self.classify = classify
        if classify:
            self.logits = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        if self.classify:
            x = self.logits(x)
        return x

def preprocess_image(image_path):
    """
    Load and preprocess the image for FaceNet.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def generate_embeddings(image_dir, model):
    """
    Generates FaceNet embeddings for all images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        model (torch.nn.Module): The FaceNet model.

    Returns:
        list: A list of NumPy arrays representing the embeddings.
    """
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    embeddings = []

    # Iterate through all images in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            img_tensor = preprocess_image(image_path)
            
            if img_tensor is None:
                continue  # Skip this image due to preprocessing error

            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                embedding = model(img_tensor)
            
            embedding_np = embedding.cpu().numpy().flatten()
            embeddings.append(embedding_np)
            print(f"Generated embedding for {filename}")

    return embeddings

def main():
    """
    Main function to execute embedding generation.
    """
    # Load the FaceNet model
    model = InceptionResnetV1(pretrained='vggface2')

    image_dir = input("Enter the path to the directory of images: ").strip()

    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        print("The specified image directory does not exist or is not a directory.")
        return

    embeddings = generate_embeddings(image_dir, model)
    if embeddings:
        print(f"\nGenerated {len(embeddings)} embeddings.")
        # Optionally, save embeddings to a file
        save_choice = input("Do you want to save the embeddings to a file? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_path = input("Enter the path for the embeddings file (e.g., embeddings.npy): ").strip()
            try:
                np.save(save_path, embeddings)
                print(f"Embeddings saved to {save_path}")
            except Exception as e:
                print(f"Error saving embeddings: {e}")
    else:
        print("No embeddings were generated.")

if __name__ == '__main__':
    main() 