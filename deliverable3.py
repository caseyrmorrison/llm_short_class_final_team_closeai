import streamlit as st
import numpy as np
import torch
import torchvision.transforms as T
from transformers import AutoModel
from transformers import AutoFeatureExtractor
from datasets import Dataset
from PIL import Image
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Assuming the model and extractor are loaded here
model_ckpt = "jafdxc/vit-base-patch16-224-finetuned-flower"
model = AutoModel.from_pretrained(model_ckpt)
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Define a transformation chain
transformation_chain = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
])

def load_dataset(images_directory):
    dataset = []
    for file in os.listdir(images_directory):
        full_path = os.path.join(images_directory, file)
        label = os.path.splitext(os.path.basename(file))[0]
        dataset.append({"image": full_path, "labels": label})
    return dataset

dataset = load_dataset("./flowers")
num_samples = min(100, len(dataset))
np.random.seed(42)
candidate_subset = np.random.choice(dataset, num_samples, replace=False)

# Extract embeddings function
def extract_embeddings(model: torch.nn.Module, image_paths):
    images = [Image.open(img_path) for img_path in image_paths]
    transformations = torch.stack([transformation_chain(img) for img in images])  
    new_batch = {"pixel_values": transformations.to(device)}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu().numpy()
    return embeddings

# Compute embeddings for all images in the subset
image_paths = [item["image"] for item in candidate_subset]
all_embeddings = extract_embeddings(model, image_paths)

# Convert embeddings to a suitable format for graph building
embeddings_dict = {item["labels"]: emb for item, emb in zip(candidate_subset, all_embeddings)}

def compute_similarity_matrix(embeddings):
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    # Convert similarity to distance
    distance_matrix = 1 / (similarity_matrix + 2)
    return distance_matrix

# Build the graph for Dijkstra's algorithm
# Assuming all_embeddings contains embeddings for all images
distance_matrix = compute_similarity_matrix(all_embeddings)

# Building the graph using distance matrix
G = nx.Graph()
for i in range(len(candidate_subset)):
    for j in range(i + 1, len(candidate_subset)):
        G.add_edge(candidate_subset[i]["labels"], candidate_subset[j]["labels"], 
                   weight=distance_matrix[i][j])

# Find shortest path function
def find_shortest_path(start, end):
    path = nx.shortest_path(G, source=start, target=end, weight='weight')
    return path

# Streamlit interface
st.title("Flower Hopper Path Finder")

# Dropdowns for selecting two images
image_options = [item["labels"] for item in candidate_subset]
image_k = st.selectbox("Select Image K", options=image_options)
image_l = st.selectbox("Select Image L", options=image_options)

if st.button("Find Path"):
    try:
        path = find_shortest_path(image_k, image_l)
        if path:
            st.write(f"Path from {image_k} to {image_l}: {path}")

            # Calculate the number of columns for the grid
            grid_size = 3  # Adjust this for a different grid size
            num_images = len(path)
            num_rows = (num_images + grid_size - 1) // grid_size  # Calculate the number of rows needed

            # Displaying images in a grid
            for i in range(num_rows):
                cols = st.columns(grid_size)
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < num_images:
                        flower_name = path[idx]
                        flower_image_path = next(item for item in candidate_subset if item["labels"] == flower_name)["image"]
                        image = Image.open(flower_image_path)
                        cols[j].image(image, caption=flower_name)
        else:
            st.write("No path found.")
    except Exception as e:
        st.write("An error occurred:", e)
