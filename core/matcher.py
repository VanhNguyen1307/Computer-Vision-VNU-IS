import os
import torch

class EmbeddingMatcher:
    def __init__(self, embeddings_dir="data/embeddings", threshold=0.55):
        """
        threshold: cosine similarity threshold
        """
        self.embeddings_dir = embeddings_dir
        self.threshold = threshold
        self.gallery = {}  # sid -> tensor (N,512)
        self.load_gallery()

    def load_gallery(self):
        self.gallery.clear()
        os.makedirs(self.embeddings_dir, exist_ok=True)

        for f in os.listdir(self.embeddings_dir):
            if not f.endswith(".pt"):
                continue
            sid = f.replace(".pt", "")
            path = os.path.join(self.embeddings_dir, f)
            embs = torch.load(path)

            # Normalize (safe)
            if embs.ndim == 1:
                embs = embs.unsqueeze(0)
            embs = embs / embs.norm(dim=1, keepdim=True)

            self.gallery[sid] = embs

    def match(self, emb):
        """
        emb: torch tensor (512,) normalized
        return: (best_id or None, best_score)
        """
        best_id, best_score = None, -1.0

        for sid, embs in self.gallery.items():
            # cosine similarity = dot product (since normalized)
            scores = torch.matmul(embs, emb)
            score = scores.max().item()
            if score > best_score:
                best_score = score
                best_id = sid

        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score
