import streamlit as st
import numpy as np
import torch
import torchvision.transforms as T
from transformers import AutoModel  # or the specific model class you're using
from transformers import AutoFeatureExtractor, AutoModel
from datasets import Dataset
from PIL import Image
import os
from tqdm.auto import tqdm
from scipy.spatial import distance

# Assuming the model and extractor are loaded here
model_ckpt = "jafdxc/vit-base-patch16-224-finetuned-flower"
model = AutoModel.from_pretrained(model_ckpt)
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size
# model.load_state_dict(torch.load('./model.pth'))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Load candidate embeddings (adjust path as needed)
# all_candidate_embeddings = torch.load('./embeddings.pth')

def load_dataset(images_directory):
      # Get paths to images
  paths_images = []
  labels = []
  for file in os.listdir(images_directory):
      paths_images.append(file)
      full_path = os.path.join(images_directory, file)
      label = os.path.splitext(os.path.basename(file))[0]
      labels.append(label)

  # Create a dataset from the paths
  dataset = Dataset.from_dict({"image": paths_images, "labels": labels})
  return dataset

dataset = load_dataset("./flowers")
num_samples = 100
seed = 42
candidate_subset = dataset.shuffle(seed=seed).select(range(num_samples))

transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(Image.open("./flowers/" + image)) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp

batch_size = 24
device = "cuda" if torch.cuda.is_available() else "cpu"
extract_fn = extract_embeddings(model.to(device))
candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=24)

all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

candidate_ids = []

for id in tqdm(range(len(candidate_subset_emb))):
    label = candidate_subset_emb[id]
    # Create a unique indentifier.
    entry = str(id) + "_" + str(label)
    candidate_ids.append(entry)

all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

def compute_distances(emb_one, emb_two):
    """Computes Euclidean distances between two sets of vectors."""
    dists = distance.cdist(emb_one, emb_two, 'euclidean')
    return dists

def fetch_similar(image, top_k=4):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    image_transformed = transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    # Compute the embedding.
    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu().numpy()

    # Compute distances with all the candidate images.
    dists = compute_distances(query_embeddings.reshape(1, -1), all_candidate_embeddings.numpy())
    dists = dists.flatten()

    # Create a mapping between the candidate image identifiers and their distances.
    distance_mapping = dict(zip(candidate_ids, dists))

    # Sort the mapping dictionary and return `top_k` candidates with smallest distances.
    distance_mapping_sorted = dict(
        sorted(distance_mapping.items(), key=lambda x: x[1]))
    id_entries = list(distance_mapping_sorted.keys())[:top_k]

    ids = [int(x.split("_")[0]) for x in id_entries]
    labels = [x.split("_")[-1] for x in id_entries]
    return ids, labels


# Streamlit interface
st.title("Flower Species Identification")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png"])
st.write("Identifying...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    sim_ids, sim_labels = fetch_similar(image, top_k=3)
    st.image(image, caption=candidate_subset_emb[sim_ids[0]]["labels"], use_column_width=True)

    st.write("Similar Flowers:")
    for id in sim_ids:
        st.image(Image.open("./flowers/" + candidate_subset_emb[id]["image"]), caption=candidate_subset_emb[id]["labels"], use_column_width=True)
