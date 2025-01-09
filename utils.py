from deepface import DeepFace
import os 
from PIL import Image
import numpy as np
import pickle
def clear_directory(directory):
    # Delete all files in the directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def load_pickle(pickle_path):
    '''loads the list of numpy array of the images '''
    with open(pickle_path, 'rb') as file:
        return pickle.load(file)
    
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

def faces_in_images(img):
    '''this function extract all the faces in the images and return the numpy array of the images '''
    obj=DeepFace.extract_faces(img,enforce_detection=False,detector_backend='retinaface')
    faces=[]
    for i in obj:
        if(i['confidence']>0.8):
            faces.append(i['face'])
    print(len(faces))
    return faces


def verify_faces(input_img,check_img):
    '''this function check for the face detected in a image with the input image and return boolean true or false '''
    response=DeepFace.verify(input_img,check_img,enforce_detection=False,model_name='VGG-Face')
    return response['verified']



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


    


