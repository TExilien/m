import os
import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
import random
import tkinter as tk
from tkinter import Canvas
import numpy as np

# -------------------------------
# 1. Setup and Load the Dataset
# -------------------------------

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load('ViT-B/32', device=device)

# Download and load the CIFAR100 dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Extract class names
class_names = cifar100.classes  # List of 100 class names
print(f"CIFAR100 Classes: {class_names}")

# -------------------------------
# 2. Build Label-to-Image Mapping
# -------------------------------

# Create a dictionary mapping each class label to the list of image indices
label_to_indices = {label.lower(): [] for label in class_names}  # Ensure labels are lowercase

for idx, (_, class_id) in enumerate(cifar100):
    label = cifar100.classes[class_id].lower()
    label_to_indices[label].append(idx)

print("Label to image indices mapping created.")

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

def get_image_for_word(word, label_to_indices, cifar100, preprocess, model, device, use_clip_fallback=True, enlarge=False):
    """
    Retrieve and display an image corresponding to the given word from the CIFAR100 dataset,
    optionally enlarging it.

    Args:
        word (str): The input word to search for.
        label_to_indices (dict): Mapping from class labels to image indices.
        cifar100 (CIFAR100): The CIFAR100 dataset object.
        preprocess (callable): Preprocessing function for images.
        model (CLIP model): The loaded CLIP model.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        use_clip_fallback (bool): Whether to use CLIP to find the closest class if exact match not found.
        enlarge (bool): Whether to enlarge the retrieved image.
        

    Returns:
        PIL.Image or None: The retrieved (and optionally enlarged) image if found, else None.
    """
    word = word.lower()
    print(f"\nSearching for the word: '{word}'")

    image = None  # Initialize image

    if word in label_to_indices and label_to_indices[word]:
        # Exact match found
        selected_index = random.choice(label_to_indices[word])
        image, class_id = cifar100[selected_index]
        print(f"Exact match found in class '{cifar100.classes[class_id]}'. Retrieving image index {selected_index}.")
    elif use_clip_fallback:
        print(f"No exact match found for '{word}'. Using CLIP to find the most similar class.")
        # Use CLIP to find the most similar class
        text = clip.tokenize([word]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Encode all class names
            class_prompts = [f"a photo of a {cls}" for cls in class_names]
            class_tokens = clip.tokenize(class_prompts).to(device)

            class_features = model.encode_text(class_tokens)
            class_features /= class_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity between input word and all class labels
            similarities = (text_features @ class_features.T).squeeze(0)  # Shape: (100,)

            # Find the class with the highest similarity
            best_match_idx = similarities.argmax().item()
            best_match_label = class_names[best_match_idx].lower()
            best_similarity = similarities[best_match_idx].item()

            print(f"Best match: '{class_names[best_match_idx]}' with similarity {best_similarity:.4f}")

            if best_match_label in label_to_indices and label_to_indices[best_match_label]:
                selected_index = random.choice(label_to_indices[best_match_label])
                image, class_id = cifar100[selected_index]
                print(f"Retrieving image from class '{cifar100.classes[class_id]}' (Index {selected_index}).")
            else:
                print(f"No images found for the matched class '{class_names[best_match_idx]}'.")
                return None
    else:
        print(f"No images found for the word: '{word}'")
        return None

    if image:
        # Optionally enlarge the image
        if enlarge:
            print("Resizing the image using interpolation.")
            image = resize_image(image, scale_factor=4, interpolation=Image.BICUBIC)
            print("Image has been enlarged.")

        # Display the image
        image.show()
        return image
    else:
        print("No image retrieved.")
        return None

# -------------------------------
# 5. Interactive Retrieval with Enlargement Options
# -------------------------------

def user(user_input):
    print("CIFAR100 Image Retrieval using CLIP")
    print("Type 'exit' to quit.\n")

    user_input = user_input.strip()
    # Ask the user if they want to enlarge the image
    enlarge_input = input("Do you want to enlarge the image? (yes/no): ").strip().lower()
    enlarge = enlarge_input in ['yes', 'y']
    pix = get_image_for_word(
            word=user_input,
            label_to_indices=label_to_indices,
            cifar100=cifar100,
            preprocess=preprocess,
            model=model,
            device=device,
            use_clip_fallback=True,  # Set to False if you want to disable CLIP fallback
            enlarge=enlarge
        )
        #Convert the image to RGB (if it's in a different format)
    pix = pix.convert("RGB")

    # Get pixel data as a NumPy array
    pixel_data = np.array(pix)
    print(pixel_data)  # This will print the array of pixel values
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

# Set up the Tkinter window and canvas
def run_drawing_app(image_pixel_data):
    root = tk.Tk()
    root.title("Shape Drawing App")
    print("Setting up the Tkinter window...")
    
    # Create a canvas for drawing
    canvas = Canvas(root, width=224, height=224, bg="white")
    canvas.pack()
    
    # Draw the image as shapes on the canvas
    draw_image_as_shapes(canvas, image_pixel_data)
    
    root.mainloop()



