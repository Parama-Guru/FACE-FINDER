import os 
import pickle
from PIL import Image
import numpy as np 
from deepface import DeepFace
import matplotlib.pyplot as plt

image_directory="photos"
output_file ="img_dictionaries.pkl"
output={}
img_list=[]
face_list=[]


for filename in os.listdir(image_directory):
    if filename.endswith((".png",".jpeg",".jpg")):
        file_path=os.path.join(image_directory,filename)
        try:
            with Image.open(file_path) as img:
                img=img.convert("RGB")
                image_array=np.array(img)
                img_list.append(image_array)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
print(f"processed images {len(img_list)}")
output = {}
for index, file in enumerate(img_list):
    print(f"processing {index + 1}")
    obj = DeepFace.extract_faces(file, detector_backend="retinaface", enforce_detection=False)
    print(f"no of faces {len(obj)},type{type(obj)}")
    faces_data = []
    for i, j in enumerate(obj):
        print(type(j))
        print(f"processing the face {i+1}")
        if j["confidence"] > 0.95:
            faces_data.append(j["face"])
            plt.imshow(j['face'])
            plt.show()
    if faces_data:
        output[index + 1] = {"image": file, "faces": faces_data}



with open(output_file,'wb') as file :
    pickle.dump(output,file)

print(f"Processed {len(output)} images and saved to {output_file}.")


