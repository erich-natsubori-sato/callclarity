from sentence_transformers import SentenceTransformer
from callclarity.utils.device import get_torch_device

class TextEmbedder:
    
    def __init__(self, texts):
        self.texts = texts
        self.embeddings = []

    def embed(self):
        model = SentenceTransformer('all-distilroberta-v1', device = get_torch_device())
        embeddings = model.encode(self.texts, batch_size=128, show_progress_bar=True)
        self.embeddings = embeddings
        return embeddings