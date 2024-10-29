import os
import clip
from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Toplevel, Button
from collections import defaultdict
import time

# -------------------------------
# Setup and Load the CLIP Model
# -------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# -------------------------------
# Load Custom Images and Class Names
# -------------------------------

image_dir = r"C:\Users\mrtyl\OneDrive\Desktop\testImages"  # Replace with the actual path
class_names = ["apple", "banana", "cat", "water Bottle", "airplane", "bird", "car", "deer", "horse"]
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
label_to_indices = defaultdict(list)

# Load images and map them to correct labels based on file names
images = []
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # Match label based on filename; skip if no match is found
    label = next((name for name in class_names if name.lower() in image_file.lower()), None)
    if label:
        label_to_indices[label].append(idx)  # Map image index to the correct label
        images.append(preprocess(image).unsqueeze(0).to(device))

# -------------------------------
# Updated Retrieval Function to Filter by Label
# -------------------------------

def get_images_for_word(word, images, label_to_indices):
    word = word.lower()

    if word in label_to_indices:
        indices = label_to_indices[word]
        selected_images = [images[idx] for idx in indices]
        return selected_images
    else:
        print(f"No images found for the word '{word}'.")
        return []

# -------------------------------
# Pop-Up Gallery Window
# -------------------------------

def open_gallery_window(canvas, word):
    selected_images = get_images_for_word(word, images, label_to_indices)

    if not selected_images:
        print(f"No images found for the word '{word}'.")
        return

    # Create a pop-up window to show the gallery
    gallery_window = Toplevel()
    gallery_window.title(f"Gallery for '{word}'")
    
    def on_image_click(selected_image_tensor):
        gallery_window.destroy()
        
        # Display each new image at an offset to avoid overlap
        x_offset = 20 * (len(displayed_images) % 5)  # Horizontal offset
        y_offset = 20 * (len(displayed_images) // 5)  # Vertical offset
        display_image_on_canvas(canvas, selected_image_tensor, x=x_offset, y=y_offset)
    
    for i, image_tensor in enumerate(selected_images):
        unnormalize = transforms.Normalize(
            mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
            std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
        image = unnormalize(image_tensor.squeeze(0))
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        thumbnail = image_pil.resize((100, 100), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(thumbnail)

        button = Button(gallery_window, image=image_tk, command=lambda img=image_tensor: on_image_click(img))
        button.image = image_tk  # Keep a reference to prevent garbage collection
        button.grid(row=i // 3, column=i % 3, padx=10, pady=10)

# -------------------------------
# Display Selected Image on Main Canvas
# -------------------------------
displayed_images = []
def display_image_on_canvas(canvas, image_tensor, x=0, y=0):
    # Unnormalize and prepare the image for display
    unnormalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
    image = unnormalize(image_tensor.squeeze(0))
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Resize for canvas display
    resized_image = image_pil.resize((150, 150), Image.LANCZOS)
    image_tk = ImageTk.PhotoImage(resized_image)
    displayed_images.append(image_tk)  # Save reference to prevent garbage collection
    
    # Place the image on the canvas
    canvas.create_image(x, y, anchor='nw', image=image_tk)
