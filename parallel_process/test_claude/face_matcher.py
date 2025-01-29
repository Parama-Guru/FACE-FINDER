# face_matcher.py
import requests
import time
from threading import Thread
import logging
from multiprocessing import Queue
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe set for storing matched IDs
matched_ids = set()
matched_ids_lock = threading.Lock()

def compare_faces(reference_face, queue_face, image_path):
    """Compare two faces using the API"""
    try:
        response = requests.post(
            'http://localhost:5000/compare_faces',
            json={
                'face1': reference_face,
                'face2': queue_face
            }
        )
        
        if response.status_code == 200:
            similarity = response.json()['similarity']
            # If similarity is above threshold, add to matched set
            if similarity > 0.7:  # Adjust threshold as needed
                with matched_ids_lock:
                    matched_ids.add(image_path)
                logger.info(f"Match found: {image_path}")
                
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")

def process_queue(reference_face, face_queue, num_threads=4):
    """Process faces from queue using multiple threads"""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while True:
            try:
                # Get face from queue with timeout
                queue_face, image_path = face_queue.get(timeout=1)
                
                # Submit comparison task
                executor.submit(compare_faces, reference_face, queue_face, image_path)
                
            except queue.Empty:
                # Check if detection is still running
                # You might want to implement a proper shutdown mechanism
                time.sleep(1)
                continue
                
            except Exception as e:
                logger.error(f"Error processing queue: {str(e)}")

if __name__ == '__main__':
    # Configure based on system
    NUM_THREADS = 4
    
    # Get reference face (you need to implement this based on your needs)
    reference_face = "base64_encoded_reference_face"
    
    # Start processing queue
    process_queue(reference_face, face_queue, NUM_THREADS)