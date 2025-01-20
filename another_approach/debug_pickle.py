import os
from PIL import Image
import numpy as np
import pickle

# Specify the path to the pickle file
pickle_file = "/Users/guru/Hackathon/aXtr_labs/another_approach/img_dictionaries.pkl"  # Use the pickle file from preprocess.py

# Open the pickle file in read-binary mode and load the list
with open(pickle_file, 'rb') as file:
    image_dict = pickle.load(file)

# Specify the output directories
images_directory = 'debug_images'
faces_directory = 'debug_faces'

# Create directories if they don't exist
os.makedirs(images_directory, exist_ok=True)
os.makedirs(faces_directory, exist_ok=True)


# Loop through each image in the dictionary and save it as an image
for image_id, image_data in image_dict.items():
    try:
        img_array = image_data['image']
        # Convert the NumPy array to a Pillow Image
        image = Image.fromarray(np.uint8(img_array))  # Ensure the array is in uint8 format
        # Define the output file path
        output_file = os.path.join(images_directory, f'image_{image_id}.png')
        # Save the image
        image.save(output_file)
        print(f"Saved image: {output_file}")

        '''#Save faces
        for i, face in enumerate(image_data['faces']):
            face_image = Image.fromarray(np.uint8(face))
            face_image = face_image[:, :, ::-1] # Convert BGR to RGB
            face_output_file = os.path.join(faces_directory, f'image_{image_id}_face_{i+1}.png')
            face_image.save(face_output_file)
            print(f"Saved face: {face_output_file}")'''

    except Exception as e:
        print(f"Error processing image {image_id}: {e}")

print(f"All images have been saved to {images_directory}.")
print(f"All faces have been saved to {faces_directory}.")