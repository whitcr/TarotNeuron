# config/settings.py
import torch

EMBEDDING_DIM = 64
NUM_TRAKTOVOK = 360
SEED = 42
VOCAB_SIZE = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")