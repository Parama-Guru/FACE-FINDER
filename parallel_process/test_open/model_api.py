# model_api.py - Handles API for RetinaFace & FaceNet, model downloading

from flask import Flask, request, jsonify
import torch
import os
from facenet_pytorch import InceptionResnetV1
from retinaface import RetinaFace
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Ensure models are stored locally
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load RetinaFace model
def load_retinaface():
    return RetinaFace()

# Load FaceNet model
def load_facenet():
    return InceptionResnetV1(pretrained='vggface2').eval()

# Initialize models
retinaface_model = load_retinaface()
facenet_model = load_facenet()

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)
    faces = retinaface_model.detect_faces(image)
    return jsonify(faces)

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    data = request.get_json()
    embedding1 = torch.tensor(data['embedding1'])
    embedding2 = torch.tensor(data['embedding2'])
    distance = torch.dist(embedding1, embedding2).item()
    match = distance < 0.6  # Threshold for similarity
    return jsonify({"match": match, "distance": distance})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
