# api_server.py
import torch
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
from flask import Flask, request, jsonify
import numpy as np
import os
import cv2
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import traceback
import warnings
import torch.cuda

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
retina_model = None
facenet_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model_directory():
    """Create models directory if it doesn't exist"""
    try:
        if not os.path.exists('models'):
            os.makedirs('models')
            logger.info("Created models directory")
    except Exception as e:
        logger.error(f"Error creating models directory: {str(e)}")
        raise

def download_and_save_models():
    """Download and save models to the models directory"""
    try:
        create_model_directory()
        
        # Save RetinaFace model
        retina_path = os.path.join('models', 'retinaface_model.pth')
        if not os.path.exists(retina_path):
            logger.info("Downloading RetinaFace model...")
            retina_model = RetinaFace(quality="normal")
            torch.save(retina_model.state_dict(), retina_path)
            logger.info("RetinaFace model saved successfully")
        
        # Save FaceNet model
        facenet_path = os.path.join('models', 'facenet_model.pth')
        if not os.path.exists(facenet_path):
            logger.info("Downloading FaceNet model...")
            facenet_model = InceptionResnetV1(pretrained='vggface2')
            torch.save(facenet_model.state_dict(), facenet_path)
            logger.info("FaceNet model saved successfully")
            
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}\n{traceback.format_exc()}")
        raise

def load_models():
    """Load models from saved files"""
    try:
        global retina_model, facenet_model
        
        # Ensure models are downloaded
        download_and_save_models()
        
        logger.info("Loading models...")
        
        # Load RetinaFace model
        retina_path = os.path.join('models', 'retinaface_model.pth')
        retina_model = RetinaFace(quality="normal")
        retina_model.load_state_dict(torch.load(retina_path))
        retina_model.to(device)
        retina_model.eval()
        
        # Load FaceNet model
        facenet_path = os.path.join('models', 'facenet_model.pth')
        facenet_model = InceptionResnetV1(pretrained='vggface2')
        facenet_model.load_state_dict(torch.load(facenet_path))
        facenet_model.to(device)
        facenet_model.eval()
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
        raise

def preprocess_image(image_data):
    """Preprocess image data for model input"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np, image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": retina_model is not None and facenet_model is not None
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"error": "Health check failed"}), 500

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Endpoint for face detection using RetinaFace"""
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image provided"}), 400
        
        # Preprocess image
        image_np, _ = preprocess_image(request.json['image'])
        
        # Detect faces
        with torch.no_grad():
            faces = retina_model.detect(image_np)
        
        # Process results
        if faces is None:
            return jsonify({"faces": []})
            
        face_images = []
        for box in faces[0]:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = image_np[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)
            
            # Convert face to base64
            buffered = io.BytesIO()
            face_pil.save(buffered, format="JPEG")
            face_base64 = base64.b64encode(buffered.getvalue()).decode()
            face_images.append(face_base64)
        
        logger.info(f"Detected {len(face_images)} faces")
        return jsonify({"faces": face_images})
    
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Face detection failed"}), 500

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    """Endpoint for face comparison using FaceNet"""
    try:
        if 'face1' not in request.json or 'face2' not in request.json:
            return jsonify({"error": "Both faces must be provided"}), 400
        
        # Preprocess faces
        face1_np, face1_pil = preprocess_image(request.json['face1'])
        face2_np, face2_pil = preprocess_image(request.json['face2'])
        
        # Resize faces to required size (160x160)
        face1_pil = face1_pil.resize((160, 160))
        face2_pil = face2_pil.resize((160, 160))
        
        # Convert to tensors
        face1_tensor = torch.from_numpy(np.array(face1_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        face2_tensor = torch.from_numpy(np.array(face2_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Normalize tensors
        face1_tensor = (face1_tensor - 127.5) / 128.0
        face2_tensor = (face2_tensor - 127.5) / 128.0
        
        # Get embeddings
        with torch.no_grad():
            embedding1 = facenet_model(face1_tensor)
            embedding2 = facenet_model(face2_tensor)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        
        logger.info(f"Face comparison completed. Similarity: {float(similarity)}")
        return jsonify({
            "similarity": float(similarity),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Face comparison failed"}), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500

if __name__ == '__main__':
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Configure server
    port = int(os.environ.get('PORT', 5000))
    processes = int(os.environ.get('PROCESSES', 3))
    
    # Log configuration
    logger.info(f"Starting server on port {port} with {processes} processes")
    logger.info(f"Using device: {device}")
    
    # Initialize models
    load_models()  # Ensure models are loaded before running
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=port,
        processes=processes,
        debug=False,
        threaded=True
    )