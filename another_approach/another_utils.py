import pickle
import os
from PIL import Image
import numpy as np 
from deepface import DeepFace
def function1 (dict,input):
    matches=[]
    for index ,obj in dict.items():
        print(f'processing the image{index}')
        for face in obj['faces']:
            response=DeepFace.verify(face,input,enforce_detection=False,model_name="VGG-Face")
            if response['verified'] == True:
                matches.append(obj['image'])
                break
    return matches 
def load_directory(pickle_path):
    '''loads the dictinoaries'''
    with open(pickle_path, 'rb') as file:
        return pickle.load(file)
    
def clear_directory(path):
    # Delete all files in the directory
    for directory in path:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def store_images(numpy_arrays,output_directory):
    clear_directory(output_directory)
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
    return True
def image_to_numpy_to_face(image_path):
    """
    Converts an image to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: NumPy array representation of the image.
    """
    try:
        # Open the image using Pillow
        img = Image.open(image_path)
        
        # Ensure the image is in RGB format
        img = img.convert('RGB')
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        
        output=DeepFace.extract_faces(img_array,detector_backend='retinaface',enforce_detection=False)
        print(len(output))
        return output[0]["face"]
    except Exception as e:
        print(f"Error converting image to numpy array: {e}")
        return None