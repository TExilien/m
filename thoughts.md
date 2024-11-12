Ok so I have the CLIP which can take an image and basically guess the word that corresponds to it.
I also am able to take in a microphone input and using google speech recognition it will match to the text equivalent.
Now I can also then match that word against all words in the database of clip and find a corresponding image to that word.
Once I have the correct Image I now want to recreate that image in the drawing software.


going back but heres the final non working product:

import torch
import clip
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Transform to resize images to 224x224 and normalize them as required by CLIP
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Default enlargement to 224x224
    transforms.ToTensor(),          # Convert PIL image to tensor
    preprocess.transforms[1]        # Normalize the tensor as required by CLIP
])

# Load CIFAR-100 dataset
from torchvision.datasets import CIFAR100
cifar100_data = CIFAR100(root='./data', download=True)

# Encode all images in CIFAR-100 dataset using CLIP
def encode_images(images):
    image_features = []
    with torch.no_grad():
        for image in images:
            image_tensor = transform(image).unsqueeze(0).to(device)  # Convert image to tensor and unsqueeze
            image_features.append(model.encode_image(image_tensor))
    return torch.cat(image_features).cpu().numpy()

# Extract the image features
images = [Image.fromarray(img) for img in cifar100_data.data]
image_features = encode_images(images)

# Encode user-provided text
def encode_text(text):
    text_inputs = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features.cpu().numpy()

# Find the most similar images to the input text
def find_similar_images(text, image_features, top_k=1):
    text_features = encode_text(text)
    similarities = cosine_similarity(text_features, image_features)
    best_match = np.argmax(similarities[0])
    return best_match, similarities[0][best_match]

# Get the pixel data of the top image
def get_image_pixels(index):
    image = Image.fromarray(cifar100_data.data[index])
    image = image.resize((224, 224))  # Ensure the image is enlarged
    return np.array(image)

# Example usage
def test_clip_with_cifar100(text_prompt):
    print(f"Finding similar images for: {text_prompt}")
    best_match, similarity_score = find_similar_images(text_prompt, image_features, top_k=1)
    print(f"Best match index: {best_match}, Similarity score: {similarity_score}")
    image_pixels = get_image_pixels(best_match)
    #return image_pixels

# Retrieve image pixel data for a text prompt
text_prompt = "Apple"
retrieved_image_pixels = test_clip_with_cifar100(text_prompt)
print("Image pixels retrieved.")

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

# Run the drawing app with the retrieved image pixel data
run_drawing_app(retrieved_image_pixels)

The last version that works where you can get the image but not the pixilated version:

import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
import random

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

def main():
    print("CIFAR100 Image Retrieval using CLIP")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a word to retrieve an image: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        elif user_input == '':
            print("Please enter a valid word.")
            continue

        # Ask the user if they want to enlarge the image
        enlarge_input = input("Do you want to enlarge the image? (yes/no): ").strip().lower()
        enlarge = enlarge_input in ['yes', 'y']

        
        get_image_for_word(
            word=user_input,
            label_to_indices=label_to_indices,
            cifar100=cifar100,
            preprocess=preprocess,
            model=model,
            device=device,
            use_clip_fallback=True,  # Set to False if you want to disable CLIP fallback
            enlarge=enlarge
        )

if __name__ == "__main__":
    main()



The above updated to get pixel data:

import os
import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
import random

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

def main():
    print("CIFAR100 Image Retrieval using CLIP")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a word to retrieve an image: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        elif user_input == '':
            print("Please enter a valid word.")
            continue

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

if __name__ == "__main__":
    main()



Need to change this to my own images 

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
            image = resize_image(image, scale_factor=24, interpolation=Image.BICUBIC)
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



If I do the image being magically brought in run into the pixel issue so without lasso tool takes apart image:


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

# Store references to images to prevent garbage collection
displayed_images = []

def display_image_with_fade(canvas, image_tensor, root, x=0, y=0, delay=50):
    """Display an image on the canvas with a fade-in effect from top to bottom."""
    # Unnormalize and prepare the image
    unnormalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
    image = unnormalize(image_tensor.squeeze(0))
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Resize for canvas display
    resized_image = image_pil.resize((150, 150), Image.LANCZOS)
    img_width, img_height = resized_image.size

    # Divide the image into slices (rows of pixels)
    slices = []
    for i in range(img_height):
        slice_image = resized_image.crop((0, i, img_width, i + 1))  # Crop one row at a time
        slice_tk = ImageTk.PhotoImage(slice_image)
        slices.append(slice_tk)

    displayed_images.append(slices)  # Prevent garbage collection

    def animate_image(slice_idx=0):
        if slice_idx < img_height:
            # Display each row slice in sequence to create the fade-in effect
            canvas.create_image(x, y + slice_idx, anchor='nw', image=slices[slice_idx])
            root.after(delay, animate_image, slice_idx + 1)  # Schedule the next slice with delay

    animate_image()  # Start the animation

# Gallery window with animated fade-in effect for selected images
def open_gallery_window(canvas, word,root):
    selected_images = get_images_for_word(word, images, label_to_indices)

    if not selected_images:
        print(f"No images found for the word '{word}'.")
        return

    gallery_window = Toplevel()
    gallery_window.title(f"Gallery for '{word}'")
    
    def on_image_click(selected_image_tensor):
        gallery_window.destroy()
        
        # Display each new image with an offset to avoid overlap and a fade-in effect
        x_offset = 20 * (len(displayed_images) % 5)  # Horizontal offset
        y_offset = 20 * (len(displayed_images) // 5)  # Vertical offset
        display_image_with_fade(canvas, selected_image_tensor, root, x=x_offset, y=y_offset)
    
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


Issue now is some of the images have a white box around them so need to select ones that dont like the drawn apple


Trying to use google search - doesn't work bc cant get good pictures just get crappy ones with terrible backgrounds
import os
import clip
from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Topleva el, Button
from collections import defaultdict
import time
import requests
from io import BytesIO
from rembg import remove  # Import rembg for background removal

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

def google_search_images(query, num_results=10):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "searchType": "image",
        "num": num_results,
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    search_results = response.json().get("items", [])
    image_urls = [item["link"] for item in search_results]
    return image_urls

# -------------------------------
# Updated Retrieval Function to Use Google API with Background Removal
# -------------------------------

def get_images_for_word(word):
    try:
        search_term = f"a singular {word} PNG transparent background"
        image_urls = google_search_images(search_term, num_results=10)
        
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
    image_tk = ImageTk.PhotoImage(resized_image)
    displayed_images.append(image_tk)  # Save reference to prevent garbage collection
    
    # Place the image on the canvas
    canvas.create_image(x, y, anchor='nw', image=image_tk)



Trying to use ai and searches for everything - doesn't work cause image sizes and whatnot
import os
import clip
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Toplevel, Button
from io import BytesIO
import requests

# Setup for device and CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Pre-trained classifier for filtering
classifier = models.resnet18(pretrained=True).eval()  # Using ResNet-18

# Transformation for classifier
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dictionary to map common words to ImageNet-compatible labels (simplified example)
word_to_class_id = {
    "banana": 954,
    "apple": 948,
    "cat": 281,
    "dog": 243,
    # Add more mappings as needed, or use approximate labels
}
# Your Google API key and Custom Search Engine ID (CSE ID)
API_KEY = "AIzaSyAvYAyQgwtMddW1Y_gtglh-Re5DGY12S-M"
CSE_ID = "831f214a05dd14b5a"

def google_search_images(query, num_results=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "searchType": "image",
        "num": num_results,
        "imgType": "clipart",  # Use clipart to help get simple objects with transparent backgrounds
        "fileType": "png",     # Request PNGs to increase the chance of transparency
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error if the request failed
    data = response.json()

    # Extract image URLs from the response
    image_urls = [item["link"] for item in data.get("items", [])]
    return image_urls

# Function to determine if an image resembles the target object based on the provided word
def is_target_object(image, target_word):
    class_id = word_to_class_id.get(target_word.lower())
    if class_id is None:
        print(f"No ImageNet mapping found for '{target_word}'. Skipping filtering.")
        return True  # If no match is found, assume image is correct

    image_tensor = classifier_transform(image).unsqueeze(0)  # Prepare image for classifier
    with torch.no_grad():
        output = classifier(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Check if target class ID is in top 5 predictions
        return class_id in top5_catid.tolist()

# Function to retrieve and filter images based on word
def get_images_for_word(word):
    search_term = f"a singular {word} PNG transparent background"
    image_urls = google_search_images(search_term, num_results=10)
    
    selected_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            
            # Filter image by checking if it resembles the target object
            if is_target_object(img, target_word=word):
                selected_images.append(preprocess(img).unsqueeze(0).to(device))
        except Exception as e:
            print(f"Skipping an image due to error: {e}")
            continue

    return selected_images if selected_images else []

# GUI and display functions remain mostly the same
def open_gallery_window(canvas, word):
    selected_images = get_images_for_word(word)

    if not selected_images:
        print(f"No images found for the word '{word}'.")
        return

    gallery_window = Toplevel()
    gallery_window.title(f"Gallery for '{word}'")
    
    def on_image_click(selected_image_tensor):
        gallery_window.destroy()
        
        x_offset = 20 * (len(displayed_images) % 5)
        y_offset = 20 * (len(displayed_images) // 5)
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
        button.image = image_tk
        button.grid(row=i // 3, column=i % 3, padx=10, pady=10)

displayed_images = []
def display_image_on_canvas(canvas, image_tensor, x=0, y=0):
    unnormalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
    image = unnormalize(image_tensor.squeeze(0))
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8)).convert("RGBA")

    resized_image = image_pil.resize((150, 150), Image.LANCZOS)
    image_tk = ImageTk.PhotoImage(resized_image)
    displayed_images.append(image_tk)
    
    canvas.create_image(x, y, anchor='nw', image=image_tk)
