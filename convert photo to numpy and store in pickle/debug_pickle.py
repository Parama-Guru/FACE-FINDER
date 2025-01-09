import os
from PIL import Image
import numpy as np
import pickle

# Specify the path to the pickle file
pickle_file = '/Users/guru/Hackathon/aXtr_labs/image_list.pkl'  # Replace with the path to your pickle file

# Open the pickle file in read-binary mode and load the list
with open(pickle_file, 'rb') as file:
    image_list = pickle.load(file)
# List of NumPy arrays (replace with your actual list)
numpy_arrays = image_list  # Replace with your list of NumPy arrays

# Specify the output directory
output_directory = '/path/to/save/images'  # Replace with your desired directory path
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Loop through each NumPy array in the list and save it as an image
for i, img_array in enumerate(numpy_arrays):
    try:
        # Convert the NumPy array to a Pillow Image
        image = Image.fromarray(np.uint8(img_array))  # Ensure the array is in uint8 format
        # Define the output file path
        output_file = os.path.join(output_directory, f'image_{i+1}.png')
        # Save the image
        image.save(output_file)
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error saving image {i+1}: {e}")

print(f"All images have been saved to {output_directory}.")