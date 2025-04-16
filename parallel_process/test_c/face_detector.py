# face_detector.py
import os
import requests
import base64
from PIL import Image
import io
from multiprocessing import Pool, Queue, Manager
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared queue for detected faces
manager = Manager()
face_queue = manager.Queue()

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode()

def process_image(image_path):
    """Process single image and detect faces"""
    try:
        # Encode image
        encoded_image = encode_image(image_path)
        
        # Send to API
        response = requests.post(
            'http://localhost:5000/detect_faces',
            json={'image': encoded_image}
        )
        
        if response.status_code == 200:
            faces = response.json()['faces']
            # Put faces and image path in queue
            for face in faces:
                face_queue.put((face, image_path))
            logger.info(f"Processed {image_path}: Found {len(faces)} faces")
        else:
            logger.error(f"Failed to process {image_path}: {response.text}")
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def process_directory(directory_path, num_processes=4, num_threads=2):
    """Process all images in directory using multiprocessing and multithreading"""
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create process pool
    with Pool(processes=num_processes) as pool:
        # Create thread pools within each process
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all images for processing
            futures = [executor.submit(process_image, image_path) 
                      for image_path in image_files]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    logger.info("Finished processing all images")

if __name__ == '__main__':
    # Configure based on system
    NUM_PROCESSES = 3  # Leaving some cores for other tasks
    NUM_THREADS = 2    # Threads per process
    
    # Start processing
    process_directory('path/to/your/images', NUM_PROCESSES, NUM_THREADS)