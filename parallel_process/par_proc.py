import os
import multiprocessing
from deepface import DeepFace
import shutil
import time

def copy_image(source_path, destination_dir):
    try:
        # Ensure the destination directory exists
        os.makedirs(destination_dir, exist_ok=True)
        
        # Construct the destination path
        destination_path = os.path.join(destination_dir, os.path.basename(source_path))
        
        # Copy the image to the destination directory
        shutil.copy(source_path, destination_path)
        
        print(f"Image copied successfully to: {destination_path}")
    except Exception as e:
        print(f"Error copying image: {e}")


def process_image(input_image_path, image_path, output_dir):
    try:
        print(f"Processing image: {image_path}")
        start_time = time.time()
        obj=DeepFace.extract_faces(img_path=image_path,enforce_detection=False,detector_backend="opencv")
        input=DeepFace.extract_faces(img_path=input_image_path,detector_backend="opencv",enforce_detection=False)
        for face in obj:
            response=DeepFace.verify(input[0]['face'],face['face'],model_name='Facenet512',enforce_detection=False)
            if response['verified'] == True:
                copy_image(image_path,output_dir)
                end_time = time.time()
                print(f"Image {image_path} verification successful. Time taken: {end_time - start_time:.4f} seconds")
                return True
        end_time = time.time()
        print(f"Image {image_path} verification unsuccessful. Time taken: {end_time - start_time:.4f} seconds")
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_chunk(args):
    input_image_path, image_paths, output_dir = args
    print(f"Processing chunk of {len(image_paths)} images")
    results = [process_image(input_image_path, image_path, output_dir) for image_path in image_paths]
    return results


def parallel_process_images(input_image_path, image_dir, output_dir):
    # Dynamically get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather all image file paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    if not image_paths:
        print("No images found in the directory.")
        return

    # Split the image paths into chunks
    num_images = len(image_paths)
    chunk_size = max(1, num_images // num_cores)
    chunks = [image_paths[i:i + chunk_size] for i in range(0, num_images, chunk_size)]

    print(f"Number of cores: {num_cores}")
    print(f"Number of images: {num_images}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {len(chunks)}")

    start_time = time.time()
    # Parallel processing using a multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, [(input_image_path, chunk, output_dir) for chunk in chunks])

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.4f} seconds")

    # Check for errors
    all_success = all(all(chunk_results) for chunk_results in results)
    if not all_success:
        print("Some images failed to process.")
    else:
        print("All images processed successfully!")


# Example usage
if __name__ == "__main__":
    image_directory = "/Users/guru/Desktop/unwanted/axtr photo"  # REPLACE WITH YOUR IMAGE DIRECTORY
    output_directory = "processed_images"         # REPLACE WITH YOUR OUTPUT DIRECTORY
    input_image = "/Users/guru/Desktop/guru.jpeg" # REPLACE WITH YOUR INPUT IMAGE PATH

    print(f"Input image: {input_image}")
    print(f"Image directory: {image_directory}")
    print(f"Output directory: {output_directory}")

    parallel_process_images(input_image, image_directory, 
                            output_directory)