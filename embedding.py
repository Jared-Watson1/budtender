from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import (
    get_embedding,
    cosine_similarity,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
import dotenv
import os
import openai
import pandas as pd
import pickle
import tiktoken
import numpy as np

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
top_n = 1000
input_datapath = "data/wiki_data.csv"


def createEmbeddings(inputFile=input_datapath):

    df = pd.read_csv(inputFile)
    df = df[["title", "heading", "content", "tokens"]]
    df = df.dropna()
    df["combined"] = (
        "Heading: " + df.heading.str.strip() + "; Content: " + df.content.str.strip()
    )

    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit reviews that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    print(len(df))
    df["embedding"] = df.combined.apply(
        lambda x: get_embedding(x, engine=embedding_model))
    df.to_csv("data/pm_embeddings.csv")


def filter(df, inp, n=3, pprint=False):
    df.head()
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    input_embedding = get_embedding(
        inp,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, input_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    if pprint:
        for r in results:
            print(r[:200])
            print()

    return pd.array(results).astype(str)

# emb = pd.read_csv("data/pm_embeddings.csv")
# createEmbeddings()
# print(filter(emb, "cardiac arrest"))
# context = filter(emb, "cardiac arrest").astype(str)
