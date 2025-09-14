from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load once (outside function so it doesn’t reload each call)
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text: str):
    """Return a normalized embedding vector for given text using BGE-small-en-v1.5."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0]  # CLS token
    return F.normalize(embeddings, p=2, dim=1)[0].tolist()  # convert tensor → list


v1 = embed("I love playing football.")
v2 = embed("Soccer is my favorite sport.")


print("Vector length:", len(v1))
print(
    "Cosine similarity:",
    torch.nn.functional.cosine_similarity(
        torch.tensor(v1).unsqueeze(0), torch.tensor(v2).unsqueeze(0)
    ).item(),
)

