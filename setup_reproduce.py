import os
from PIL import Image

def create_dummy_images():
    images_dir = "temp_pdf_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Create 3 dummy images
    for i in range(3):
        img = Image.new('RGB', (100, 100), color = (73, 109, 137))
        img.save(f"{images_dir}/page_{i}.png")
    
    print(f"Created 3 dummy images in {images_dir}")

if __name__ == "__main__":
    create_dummy_images()
