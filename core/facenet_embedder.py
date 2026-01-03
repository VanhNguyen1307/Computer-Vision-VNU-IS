import torch
import cv2
from facenet_pytorch import InceptionResnetV1

class FaceNetEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def preprocess(self, face_img):
        """
        face_img: BGR image (OpenCV)
        return: torch tensor (1,3,160,160)
        """
        face_img = cv2.resize(face_img, (160, 160))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_img).permute(2, 0, 1).float() / 255.0
        return face_tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed(self, face_img):
        """
        return: normalized embedding vector (512,)
        """
        x = self.preprocess(face_img)
        emb = self.model(x).squeeze(0)
        emb = emb / emb.norm()  # normalize
        return emb.detach().cpu()
