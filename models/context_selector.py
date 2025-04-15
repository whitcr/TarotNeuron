import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextSelector(nn.Module):
    def __init__(self, card_names: list[str], embedding_dim: int = 64):
        """
        Инициализация механизма выбора контекста с использованием attention.

        :param card_names: Список всех карт, участвующих в раскладе
        :param embedding_dim: Размерность эмбеддингов карт
        """
        super(ContextSelector, self).__init__()
        self.card_names = card_names
        self.embedding_dim = embedding_dim

        # Простой attention-механизм между картами
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, target_card: str, all_embeddings: dict[str, torch.Tensor]):
        """
        Выбирает контекст для target_card на основе других карт.

        :param target_card: Название карты, для которой ищем трактовку
        :param all_embeddings: Словарь, содержащий эмбеддинги всех карт расклада
        :return: Сгенерированный контекстный вектор
        """
        # Получаем эмбеддинг целевой карты
        queries = self.query_proj(all_embeddings[target_card].unsqueeze(0))  # (1, D)

        keys = []
        values = []
        for name, emb in all_embeddings.items():
            if name != target_card:  # Исключаем target_card
                keys.append(self.key_proj(emb))
                values.append(self.value_proj(emb))

        # Составляем ключи и значения для всех карт
        keys = torch.stack(keys)  # (N-1, D)
        values = torch.stack(values)  # (N-1, D)

        # Считаем attention-оценки для всех карт (кроме target_card)
        attn_scores = torch.matmul(queries, keys.T) / (self.embedding_dim ** 0.5)  # (1, N-1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (1, N-1)

        # Генерируем итоговый контекстный вектор для целевой карты
        context_vector = torch.matmul(attn_weights, values)  # (1, D)
        return context_vector.squeeze(0)  # Возвращаем вектор без лишнего измерения

    def select_context(self, input_contexts: dict[str, torch.Tensor]):
        """
        Process all cards in the reading to generate context vectors.

        Args:
            input_contexts: Dictionary mapping card names to their embeddings

        Returns:
            Dictionary of context vectors for each card
        """
        context_vectors = {}
        for target_card in input_contexts:
            context_vectors[target_card] = self.forward(target_card, input_contexts)
        return context_vectors
