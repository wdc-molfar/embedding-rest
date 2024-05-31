# download.py

from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
import joblib
import numpy as np
import os


# Download model
def download_model():
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                                revision='02d96723811f4bb77a80857da07eda78c1549a4d',
                                trust_remote_code=True)
    return model


# Save model
def save_model(model):
    model_dir = "src/models/nomic-embed"  # os.environ.get('NOMIC_RESOURCES_DIR')
    model.save(model_dir)
    return "Model saved successfully!"


# Transform model
def transform_umap_model(model, umap_model_name, hyper_params: dict):
    # Load data from a text file nomic_emb.csv
    x_nom = np.genfromtxt('src/dataset/nomic_emb.csv', delimiter=",")

    # umap model
    umap_model = UMAP(**hyper_params)
    umap_model.fit_transform(x_nom)
    joblib.dump(umap_model, umap_model_name)


# Open umap models
def open_umap_model(umap_model_name):
    umap_model = joblib.load(open(umap_model_name, 'rb'))
    return umap_model

# Download and Save model, Transform and Save models
def main_download_transform(params) -> None:
    # Download model
    model = download_model()

    # Save model
    save_model(model)

    # Create dir for transformed models
    os.makedirs("src/models/umap-models/", exist_ok=True)

    # Transform models
    # umap-cluster model
    umap_cluster_model_name = 'src/models/umap-models/umap-cluster.joblib'
    params["n_components"] = 8
    transform_umap_model(model, umap_cluster_model_name, params)
    # umap-display model
    umap_display_model_name = 'src/models/umap-models/umap-display.joblib'
    params["n_components"] = 2
    transform_umap_model(model, umap_display_model_name, params)


# Setting hyperparameters
hyper_params = {"n_components": 8,
                "n_neighbors": 15,
                "random_state": 42,
                "min_dist": 0.05,
                "metric": 'cosine',
                "n_jobs": 1}

if __name__ == '__main__':
    main_download_transform(hyper_params)
