import os
from PIL import Image
import numpy as np
from retinaface import RetinaFace

def extract_faces(image_path):
    # Load image
    img = Image.open(image_path)
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img_array)

    if faces:
        # Create the 'images' directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
        
        for i, face_data in enumerate(faces.values()):
            # Extract face coordinates
            x1, y1, x2, y2 = [int(coord) for coord in face_data['facial_area']]
            
            # Crop face from image
            face_img = img.crop((x1, y1, x2, y2))
            
            # Save the face image in the 'images' directory
            output_path = os.path.join('images', f'face_{i}.jpg')
            face_img.save(output_path)
            print(f"Extracted {output_path}")
    else:
        print("No faces detected.")

def main():
    image_path = "/Users/guru/proj-git/Face-Finder/parallel_process/test/test.jpg"
    
    if os.path.exists(image_path):
        extract_faces(image_path)
    else:
        print("The specified image path does not exist.")

if __name__ == '__main__':
    main()