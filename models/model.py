# model.py — модель для расклада Таро

import torch
import torch.nn as nn
from models.context_selector import ContextSelector
from models.tarot_neuron import TarotNeuronNetwork
from models.decoder import TarotInterpreter

class TarotModel(nn.Module):
    def __init__(self, card_names: list[str], embedding_dim: int = 64, num_tractovki: int = 500, hidden_dim: int = 128, vocab_size: int = 10000):
        super().__init__()
        self.card_names = card_names
        self.embedding_dim = embedding_dim
        self.num_tractovki = num_tractovki

        # Инициализация компонентов модели
        self.neuron_network = TarotNeuronNetwork(card_names, embedding_dim, num_tractovki)
        self.context_selector = ContextSelector(embedding_dim)
        self.decoder = TarotInterpreter(embedding_dim, hidden_dim, vocab_size)

    def forward(self, input_contexts: dict[str, torch.Tensor], target_card: str):
        # Получаем трактовки для всех карт
        outputs = self.neuron_network(input_contexts)

        # Выбираем контекст для целевой карты
        target_context = self.context_selector(target_card, input_contexts)

        # Генерируем трактовку для целевой карты
        target_tractovka = outputs[target_card][0]
        generated_text = self.decoder.generate_text(target_tractovka, target_context)

        return generated_text
