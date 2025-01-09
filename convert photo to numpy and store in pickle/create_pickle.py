import os
import pickle
import numpy as np
from PIL import Image

# Define the directory containing the images
image_directory = "/Users/guru/Hackathon/aXtr_labs/photos"  # Replace with the path to your images directory
output_file = "/Users/guru/Hackathon/aXtr_labs/image_list.pkl"  # Replace with the path to save the pickle file

# Initialize an empty list to store image arrays
image_list = []

# Loop through each file in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
        file_path = os.path.join(image_directory, filename)
        try:
            # Open the image and convert it to a NumPy array
            with Image.open(file_path) as img:
                img=img.convert('RGB')
                image_array = np.array(img)
                image_list.append(image_array)  # Append the image array to the list
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Save the list of image arrays to a pickle file
with open(output_file, 'wb') as file:
    pickle.dump(image_list, file)

print(f"Processed {len(image_list)} images and saved to {output_file}.")