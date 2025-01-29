# facenet_comparator.py - Compares extracted faces with input image using FaceNet

import multiprocessing
import requests
import torch
import numpy as np

API_URL = "http://localhost:5000/compare_faces"
INPUT_IMAGE_EMBEDDING = None  # Placeholder for input image embedding
PROCESSED_IDS = set()

# Function to compare faces using FaceNet API
def compare_face(face_embedding, image_id):
    global PROCESSED_IDS
    if image_id in PROCESSED_IDS:
        return
    response = requests.post(API_URL, json={
        "embedding1": INPUT_IMAGE_EMBEDDING,
        "embedding2": face_embedding
    })
    result = response.json()
    if result["match"]:
        PROCESSED_IDS.add(image_id)
        print(f"Match found: {image_id}")

# Worker function to process faces from queue
def worker(queue):
    while True:
        if not queue.empty():
            image_id, face_embedding = queue.get()
            compare_face(face_embedding, image_id)

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(queue,))
    process.start()
    process.join()
