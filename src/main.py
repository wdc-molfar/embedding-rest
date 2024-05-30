# main.py

# Import FastAPI
from fastapi import FastAPI
# Import BaseModel from pydantic
from pydantic import BaseModel
import sys
import traceback
import warnings
from src.models import main_download_transform, download_model, save_model, transform_umap_model, open_umap_model

import joblib
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message=r"\[W033\]", category=UserWarning)

# Download nomic-embed-text-v1 model
# model = SentenceTransformer("nomic-embed",
#                            revision='02d96723811f4bb77a80857da07eda78c1549a4d',
#                            trust_remote_code=True)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                            revision='02d96723811f4bb77a80857da07eda78c1549a4d',
                            trust_remote_code=True)

# umap-cluster model
umap_cluster_model_name = 'src/models/umap-models/umap-cluster.joblib'
# umap-display model
umap_display_model_name = 'src/models/umap-models/umap-display.joblib'

# Open umap models
reduced_cluster = open_umap_model(umap_cluster_model_name)
reduced_display = open_umap_model(umap_display_model_name)

def embedding(text):
    # Embedding
    vec = model.encode(text)

    # Reducing (cluster and display vec)
    cluster_vec = reduced_cluster.transform(vec.reshape(1, -1))
    display_vec = reduced_display.transform(vec.reshape(1, -1))

    return {"cluster_vec": cluster_vec.tolist(), "display_vec": display_vec.tolist()}


# Create subclasses defining the schema, or data shapes, you want to receive
class Item(BaseModel):
    text: str

    class Config:
        orm_mode = True


# Create a FastAPI instance
app = FastAPI()

# Setting hyperparameters
hyper_params = {"n_components": 8,
                "n_neighbors": 15,
                "random_state": 42,
                "min_dist": 0.05,
                "metric": 'cosine',
                "n_jobs": 1}


# Define a path GET-operation decorator
@app.get("/download")
async def download_transform_models():
    main_download_transform(hyper_params)
    return "Models loaded successfully!"


# Define a path POST-operation decorator
@app.post("/embedding")
# Define the path operation function
async def run_embedding(input_json: Item):
    # Get text from input JSON object
    text = input_json.text

    try:
        output = {"request": input_json, "response": embedding(text)}

    except BaseException as ex:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        output = {"error": ''}
        output['error'] += "Exception type : %s; \n" % ex_type.__name__
        output['error'] += "Exception message : %s\n" % ex_value
        output['error'] += "Exception traceback : %s\n" % "".join(
            traceback.TracebackException.from_exception(ex).format())

    return output
