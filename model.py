import os
import clip
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import random
import tkinter as tk
from tkinter import Canvas
from collections import defaultdict

# -------------------------------
# 1. Setup and Load the CLIP Model
# -------------------------------

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load('ViT-B/32', device=device)

# -------------------------------
# 2. Load Custom Images and Class Names
# -------------------------------

# Directory where your custom images are stored
image_dir = r"C:\Users\mrtyl\OneDrive\Desktop\testImages"  # Replace with the actual path

# Load your own class names if needed, or set default class names
class_names = ["apple", "banana", "cat", "water Bottle"]  # Replace with your custom class names

# Load images from the specified directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
label_to_indices = {label: [] for label in class_names}

# Preprocess and load all images into memory, map them to class names
images = []
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    label = random.choice(class_names)  # Assign a random label for this example
    label_to_indices[label].append(idx)
    images.append(preprocess(image).unsqueeze(0).to(device))

print("Images and label-to-index mapping created.")

# -------------------------------
# 3. Define Image Enlargement Functions
# -------------------------------

def resize_image(image, scale_factor=4, interpolation=Image.BICUBIC):
    """
    Resize the image using PIL's interpolation.

    Args:
        image (PIL.Image): The input image to resize.
        scale_factor (int): The factor by which to enlarge the image.
        interpolation (int): The PIL interpolation method.

    Returns:
        PIL.Image: The resized image.
    """
    original_size = image.size
    new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
    resized_image = image.resize(new_size, interpolation)
    return resized_image

# -------------------------------
# 4. Define Retrieval Function with Image Enlargement
# -------------------------------
shown_images_history = defaultdict(set)
def get_image_for_word(word, images, class_names, preprocess, model, device, top_k=3):
    word = word.lower()
    print(f"\nSearching for the word: '{word}'")

    # Step 1: Encode the input word into text features
    text = clip.tokenize([word]).to(device)

    with torch.no_grad():
        # Encode the text input to get text features
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Step 2: Encode all images to get image features
        image_features = []
        for image_tensor in images:
            with torch.no_grad():
                image_features.append(model.encode_image(image_tensor))
        image_features = torch.cat(image_features, dim=0)  # Stack all image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Step 3: Compute similarity between text features and image features
        similarities = (text_features @ image_features.T).squeeze(0)  # Shape: (number_of_images,)

       # Step 4: Get the indices of the top-k most similar images
        top_k_indices = similarities.topk(top_k).indices.tolist()

        # Filter out images that have already been shown
        available_indices = [idx for idx in top_k_indices if idx not in shown_images_history[word]]

        if not available_indices:  # If all images in the top-k have been shown, reset history
            shown_images_history[word].clear()
            available_indices = top_k_indices

        # Step 5: Randomly pick one of the available images
        selected_index = random.choice(available_indices)
        best_similarity = similarities[selected_index].item()

        # Mark the selected image as shown
        shown_images_history[word].add(selected_index)

        print(f"Selected image index: {selected_index} with similarity {best_similarity:.4f}")

        # Retrieve the selected image tensor
        best_image_tensor = images[selected_index]

    if best_image_tensor is not None and best_image_tensor.numel() > 0:
        unnormalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
        image = unnormalize(best_image_tensor.squeeze(0))  # Remove batch dimension
        image = torch.clamp(image, 0, 1)  # Ensure values are within [0, 1]
        # Convert to numpy array and then PIL image for displaying
        image_np = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert to PIL Image
        
        # Display the retrieved image
        image_pil.show()
        return image_pil
    else:
        print("No image retrieved.")
        return None




# -------------------------------
# 5. Interactive Retrieval with Enlargement Options
# -------------------------------

def user(user_input):
    print("Custom Image Retrieval using CLIP")
    print("Type 'exit' to quit.\n")

    user_input = user_input.strip()
    pix = get_image_for_word(
            word=user_input,
            images=images,
            class_names=class_names,
            preprocess=preprocess,
            model=model,
            device=device,
            top_k=3
        )
    # Convert the image to RGB (if it's in a different format)
    pix = pix.convert("RGB")

    # Get pixel data as a NumPy array
    pixel_data = np.array(pix) # This is the array of pixel values 
    return pixel_data

# Convert image pixel data to shapes
def draw_image_as_shapes(canvas, pixel_data, shape_size=5):
    rows, cols, _ = pixel_data.shape
    print(f"Drawing shapes for pixel data of size: {rows}x{cols}")
    for row in range(0, rows, shape_size):
        for col in range(0, cols, shape_size):
            r, g, b = pixel_data[row, col]
            color = f'#{r:02x}{g:02x}{b:02x}'  # Convert RGB to hex
            x0, y0 = col, row
            x1, y1 = col + shape_size, row + shape_size
            canvas.create_oval(x0, y0, x1, y1, fill=color, outline=color)
    print("Finished drawing shapes.")


