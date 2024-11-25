import os
import clip
from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Toplevel, Button
from collections import defaultdict
import requests
from io import BytesIO
from rembg import remove

# -------------------------------
# Setup and Load the CLIP Model
# -------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# -------------------------------
# Google Custom Search API Setup
# -------------------------------

API_KEY = "AIzaSyAvYAyQgwtMddW1Y_gtglh-Re5DGY12S-M"
CSE_ID = "831f214a05dd14b5a"

def google_search_images(query, num_results=12):
    search_url = "https://www.googleapis.com/customsearch/v1"
    max_results_per_request = 10  # Google API limit
    image_urls = []
    
    for start in range(1, min(num_results, 101), max_results_per_request):
        # Primary query restricted to CleanPNG
        params = {
            "key": API_KEY,
            "cx": CSE_ID,
            "q": f"{query} site:https://www.cleanpng.com/free/{query}.html",
            "searchType": "image",
            "num": min(max_results_per_request, num_results - len(image_urls)),
            "start": start,
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_results = response.json().get("items", [])
            image_urls.extend([item["link"] for item in search_results])
            
            # Stop if we have enough images
            if len(image_urls) >= num_results:
                break
            
        except Exception as e:
            print(f"Primary query failed for '{query}': {e}")
            break
    
    # Fallback to a general search if insufficient results
    if len(image_urls) < num_results:
        for start in range(1, min(num_results, 101), max_results_per_request):
            params = {
                "key": API_KEY,
                "cx": CSE_ID,
                "q": query,  # General query without site restriction
                "searchType": "image",
                "num": min(max_results_per_request, num_results - len(image_urls)),
                "start": start,
            }
            
            try:
                response = requests.get(search_url, params=params)
                response.raise_for_status()
                search_results = response.json().get("items", [])
                image_urls.extend([item["link"] for item in search_results])
                
                # Stop if we have enough images
                if len(image_urls) >= num_results:
                    break
                
            except Exception as e:
                print(f"Fallback query failed for '{query}': {e}")
                break
    
    return image_urls[:num_results]  # Return exactly the requested number of images



# -------------------------------
# Updated Retrieval Function to Use Google API with Background Removal
# -------------------------------

def get_images_for_word(word):
    try:
        image_urls = google_search_images(word)
        selected_images = []
        for url in image_urls:
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGBA")  # Ensure RGBA format
                
                # Preprocess and add to selected images
                selected_images.append(preprocess(img).unsqueeze(0).to(device))
            except Exception as e:
                print(f"Skipping an image due to error: {e}")
                continue

        return selected_images if selected_images else []
    except Exception as e:
        print(f"Failed to retrieve images for '{word}': {e}")
        return []

def remove_background(image):
    """
    Use rembg to remove the background from the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    input_data = buffered.getvalue()
    output_data = remove(input_data)
    output_image = Image.open(BytesIO(output_data))
    return output_image

# -------------------------------
# Pop-Up Gallery Window
# -------------------------------

def open_gallery_window(canvas, word):
    selected_images = get_images_for_word(word)

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
        #thumbnail = remove_background(thumbnail)
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
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8)).convert("RGBA")  # Ensure RGBA for transparency

    # Resize for canvas display
    resized_image = image_pil.resize((150, 150), Image.LANCZOS)

    # Remove the checkerboard pattern
    resized_image = remove_background(resized_image)

    # Apply to your image before displaying
    image_tk = ImageTk.PhotoImage(resized_image)
    displayed_images.append(image_tk)  # Save reference to prevent garbage collection
    
    # Place the image on the canvas
    canvas.create_image(x, y, anchor='nw', image=image_tk)