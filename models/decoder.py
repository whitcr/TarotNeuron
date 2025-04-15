# decoder.py — генератор текстов трактовок из вектора трактовки

import torch
import torch.nn as nn
import torch.nn.functional as F

class TarotInterpreter(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Преобразуем трактовку в скрытое состояние
        self.fc = nn.Linear(embedding_dim, hidden_dim)

        # Декодер на GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tractovka_vector: torch.Tensor, max_len: int = 40):
        # Начальное скрытое состояние из вектора трактовки
        h0 = self.fc(tractovka_vector).unsqueeze(0)  # (1, 1, H)

        outputs = []
        input_token = torch.zeros(1, 1, self.hidden_dim, device=tractovka_vector.device)

        for _ in range(max_len):
            out, h0 = self.gru(input_token, h0)  # out: (1, 1, H)
            token_logits = self.output(out.squeeze(1))  # (1, vocab_size)
            outputs.append(token_logits)

            input_token = out  # передаём следующее скрытое состояние

        return torch.stack(outputs, dim=1)  # (1, max_len, vocab_size)

    def generate_text(self, tractovka_vector: torch.Tensor, tokenizer, max_len: int = 40):
        logits = self.forward(tractovka_vector, max_len)
        token_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        return tokenizer.decode(token_ids)