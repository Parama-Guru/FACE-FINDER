import os
from PIL import Image
import numpy as np
from retinaface import RetinaFace
from multiprocessing import Queue

import concurrent.futures
from multiprocessing import Pool
import time

class RetinaFaceDetector:
    def __init__(self, queue):
        # Build the RetinaFace model
        self.model = RetinaFace.build_model()
        self.queue = queue

    def extract_faces(self, image_path):
        # Load image and resize it to a smaller size
        img = Image.open(image_path)
        img = img.resize((480, 360))  # Adjust the size as needed
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Detect faces using the built model with adjusted parameters
        faces = RetinaFace.detect_faces(img_array, model=self.model, threshold=0.8)

        if faces:
            for face_id, face_info in faces.items():
                face_image = self.crop_face(img_array, face_info['facial_area'])
                self.queue.put({
                    'id': image_path,
                    'face': face_image
                })
                print(f"Pushed face from {image_path} to the queue")
        else:
            print(f"No faces detected in {image_path}")

    def crop_face(self, image, facial_area):
        x1, y1, x2, y2 = facial_area
        face_image = image[y1:y2, x1:x2]
        return face_image

def process_images(image_paths):
    # Create a new instance of RetinaFaceDetector for each process
    detector = RetinaFaceDetector()
    for image_path in image_paths:
        detector.extract_faces(image_path)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(detector.extract_faces, image_paths)

def main():
    image_directory = "/Users/guru/proj-git/Face-Finder/parallel_process/test/images"
    
    if os.path.exists(image_directory):
        image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, filename))]
        
        start_time = time.time()
        
        # Use a smaller number of processes
        num_processes = 4
        pool = Pool(processes=num_processes)
        
        # Split the image paths into chunks for each process
        chunk_size = len(image_paths) // num_processes
        image_chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]
        
        # Create a shared queue
        queue = Queue()
        
        # Process the image chunks in parallel
        pool.map(process_images, image_chunks)
        
        pool.close()
        pool.join()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Face detection completed in {execution_time:.2f} seconds")
    else:
        print("The specified image directory does not exist.")

if __name__ == '__main__':
    main()
