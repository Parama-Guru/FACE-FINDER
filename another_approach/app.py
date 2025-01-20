import streamlit as st 
import atexit,shutil
import os 
from PIL import Image
from another_utils import load_directory,function1,store_images,image_to_numpy_to_face,clear_directory

output_dir="images"
upload_dir = "uploaded_images"
pickle_file="/Users/guru/Hackathon/aXtr_labs/another_approach/img_dictionaries.pkl"


for dir_path in [output_dir, upload_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def cleanup():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

atexit.register(cleanup)

def main():
    st.title("Face Recognition Search")
    
    uploaded_file = st.file_uploader("Choose an image to search for", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
            path=[upload_dir,output_dir]
            clear_directory(path)
            image = Image.open(uploaded_file)
            st.image(image, caption='Query Image', use_container_width=True)
            if st.button("Search for Matches"):

                save_path = os.path.join(upload_dir, uploaded_file.name)
                image.save(save_path)
                
                with st.spinner('Searching for matches...'):
                    input=image_to_numpy_to_face(save_path)
                    dict=load_directory(pickle_file)
                    response=function1(dict,input)
                    output=store_images(response,output_dir)
                    if output is True:
                        if os.path.exists(output_dir) and os.path.isdir(output_dir):
                            # Get the list of files in the directory
                            files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

                            if files:
                                # Sort the files alphabetically
                                files.sort()

                                # Select the first four images
                                files_to_display = files[:4]

                                # Display the images
                                st.write(f"Displaying the first {len(files_to_display)} images from '{output_dir}':")
                                cols = st.columns(2)  # Create two columns for layout
                                for idx, file_name in enumerate(files_to_display):
                                    image_path = os.path.join(output_dir, file_name)
                                    with cols[idx % 2]:
                                        st.image(image_path, caption=f"Image {idx+1}: {file_name}", use_container_width=True)
                            else:
                                st.warning(f"No images found in the directory '{output_dir}'.")
                        else:
                            st.warning("The output directory does not exist.")

                

if __name__ == "__main__":
    main() 