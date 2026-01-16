from config import TOP_K

def create_retr_csv(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    print("Retriever created with TOP_K =", TOP_K)
    return retriever