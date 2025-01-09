import streamlit as st
import os
import shutil,atexit
from PIL import Image
from services import check_matches
from utils import clear_directory,load_pickle,image_to_numpy_to_face,store_images
UPLOAD_DIR = "/Users/guru/Hackathon/aXtr_labs/uploaded_images_deepface"
PICKLE_FILE = "/Users/guru/Hackathon/aXtr_labs/image_list.pkl"
OUTPUT_DIR = "/Users/guru/Hackathon/aXtr_labs/output_deepface"

for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def cleanup():
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
atexit.register(cleanup)

def main():
    st.title("Face Recognition Search")
    
    uploaded_file = st.file_uploader("Choose an image to search for", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            clear_directory(UPLOAD_DIR)
            image = Image.open(uploaded_file)
            st.image(image, caption='Query Image', use_container_width=True)
            if st.button("Search for Matches"):

                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                image.save(save_path)
                
                with st.spinner('Searching for matches...'):
                    image_list = load_pickle(PICKLE_FILE)
                    input_image= image_to_numpy_to_face(save_path)
                    image_with_face=check_matches(image_list,input_image)
                    output=store_images(image_with_face,OUTPUT_DIR)
                    if output is True:
                        if os.path.exists(OUTPUT_DIR) and os.path.isdir(OUTPUT_DIR):
                            # Get the list of files in the directory
                            files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

                            if files:
                                # Sort the files alphabetically
                                files.sort()

                                # Select the first four images
                                files_to_display = files[:4]

                                # Display the images
                                st.write(f"Displaying the first {len(files_to_display)} images from '{OUTPUT_DIR}':")
                                cols = st.columns(2)  # Create two columns for layout
                                for idx, file_name in enumerate(files_to_display):
                                    image_path = os.path.join(OUTPUT_DIR, file_name)
                                    with cols[idx % 2]:
                                        st.image(image_path, caption=f"Image {idx+1}: {file_name}", use_container_width=True)
                            else:
                                st.warning(f"No images found in the directory '{OUTPUT_DIR}'.")
                        else:
                            st.warning("The output directory does not exist.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}{save_path}")

if __name__ == "__main__":
    main() 
