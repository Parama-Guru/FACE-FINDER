import os
from PIL import Image
import numpy as np
from retinaface import RetinaFace

import concurrent.futures
from multiprocessing import Process, cpu_count

def process_faces(image_path, faces):
    # Create the 'images' directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
    
    def save_face(i, face_data):
        # Extract face coordinates
        x1, y1, x2, y2 = [int(coord) for coord in face_data['facial_area']]
        
        # Crop face from image
        face_img = Image.open(image_path).crop((x1, y1, x2, y2))
        # Save the face image in the 'images' directory with the image path and face number
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join('images', f'{name}_face_{i}{ext}')
        face_img.save(output_path)
        print(f"Extracted {output_path}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(save_face, range(len(faces)), faces.values())

def extract_faces(image_path):
    # Load image
    img = Image.open(image_path)
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img_array)

    if faces:
        process_faces(image_path, faces)
    else:
        print("No faces detected.")

def process_images(image_paths):
    for image_path in image_paths:
        extract_faces(image_path)

def main():
    image_directory = "/Users/guru/proj-git/Face-Finder/parallel_process/test/images"
    
    if os.path.exists(image_directory):
        image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, filename))]
        
        # Use two processes
        num_processes = min(2, cpu_count())
        chunk_size = len(image_paths) // num_processes
        print(num_processes)
        print(chunk_size)
        processes = []
        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i != num_processes - 1 else len(image_paths)
            process = Process(target=process_images, args=(image_paths[start_index:end_index],))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
    else:
        print("The specified image directory does not exist.")

if __name__ == '__main__':
    main()
