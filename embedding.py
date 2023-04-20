from openai.embeddings_utils import (
    get_embedding,
    cosine_similarity,
)
import dotenv
import os
import openai
import pandas as pd
import numpy as np

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
top_n = 1000
input_datapath = "data/wiki_data.csv"

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
