# from langchain_community.embeddings import HuggingFaceEmbeddings
# from config import EMBEDDING_MODEL, FAISS_INDEX_PATH
# from langchain_community.vectorstores import FAISS
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

# def build_faiss_index_csv(split_docs, recreate=True):
#     if os.path.exists(FAISS_INDEX_PATH) and not recreate:
#         print("FAISS index exists, loading from disk.")
#         vector_store = FAISS.load_local(
#             FAISS_INDEX_PATH,
#             HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,),
#             allow_dangerous_deserialization=True,
#         )
#         return vector_store
#     print("Creating new FAISS index.")

#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     vector_store = FAISS.from_documents(split_docs, embeddings)
#     vector_store.save_local(FAISS_INDEX_PATH)
#     print("FAISS index created and saved to disk.",FAISS_INDEX_PATH)
#     return vector_store

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from config import EMBEDDING_MODEL, FAISS_INDEX_PATH
# from langchain_community.vectorstores import FAISS
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Disable all GPUs

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

# def build_faiss_index_csv(split_docs, recreate=True):
#     if os.path.exists(FAISS_INDEX_PATH) and not recreate:
#         print("FAISS index exists, loading from disk.")
#         vector_store = FAISS.load_local(
#             FAISS_INDEX_PATH,
#             HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}),
#             allow_dangerous_deserialization=True,
#         )
#         return vector_store

#     print("Creating new FAISS index.")

#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL,
#         model_kwargs={"device": "cpu"}   # Forces CPU for embeddings
#     )

#     vector_store = FAISS.from_documents(split_docs, embeddings)
#     vector_store.save_local(FAISS_INDEX_PATH)

#     print("FAISS index created and saved to disk.", FAISS_INDEX_PATH)
#     return vector_store

import os
from typing import List, Sequence
from huggingface_hub import InferenceClient
# from config import HF_TOKEN
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_TOKEN = os.environ.get("HF_TOKEN")

class HFInferenceEmbeddings:
    def __init__(self, model: str = DEFAULT_MODEL, token: str | None = None):
        self.model = model
        token = token or os.getenv("HF_TOKEN")
        if token:
            self.client = InferenceClient(token=token)
        else:
            self.client = InferenceClient()  # unauthenticated may work for public endpoints
    def _call(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self.client.embeddings(model=self.model, inputs=list(texts))
        vectors = []
        for item in resp:
            if isinstance(item, dict) and "embedding" in item:
                vectors.append(item["embedding"])
            elif isinstance(item, (list, tuple)):
                vectors.append(list(item))
            else:
                raise ValueError(f"Unexpected HF response format: {item}")
        return vectors
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)
    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]
