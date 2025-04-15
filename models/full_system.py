import torch
from models.context_selector import ContextSelector
from models.tarot_neuron import TarotNeuronNetwork
from models.decoder import TarotInterpreter

class TarotSystem:
    def __init__(self, card_names: list[str], embedding_dim: int = 64, num_tractovki: int = 500):
        # Инициализируем компоненты
        self.context_selector = ContextSelector(card_names, embedding_dim)
        self.neuron_network = TarotNeuronNetwork(card_names, embedding_dim, num_tractovki)
        self.decoder = TarotInterpreter()

    def process_reading(self, input_contexts: dict[str, torch.Tensor]):
        # Шаг 1: Получение контекста
        context_vectors = self.context_selector.select_context(input_contexts)

        # Шаг 2: Прогон нейронной сети для получения трактовок
        tractovki = self.neuron_network(context_vectors)

        # Шаг 3: Генерация трактовок с помощью декодера
        results = {}
        for card, (tractovka, idx) in tractovki.items():
            text = self.decoder.generate_text(tractovka)
            results[card] = {
                "tractovka_index": idx,
                "text": text
            }

        return results
