import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.context_selector import ContextSelector
from models.tarot_neuron import TarotNeuronNetwork
from utils.visualizer import visualize_tarot_sphere

# Параметры
card_names = ["The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", "The Hierophant", "The Lovers", "The Chariot"]
embedding_dim = 64
num_tractovki = 500
radius = 5.0

# Инициализация эмбеддингов карт
# Например, мы генерируем случайные эмбеддинги для карт
all_embeddings = {
    card_name: torch.randn(embedding_dim) for card_name in card_names
}

# Инициализация нейронной сети для карт Таро
tarot_network = TarotNeuronNetwork(card_names, embedding_dim=embedding_dim, num_tractovki=num_tractovki)

# Инициализация контекстного селектора
context_selector = ContextSelector(card_names, embedding_dim)

# Пример использования ContextSelector для выбора контекста для карты
target_card = "The Fool"
context_vector = context_selector(target_card, all_embeddings)

# Визуализация 3D-сферы трактовок для карты Таро
# Для этого используем активную трактовку, индекс которой выбрал ContextSelector
active_index = 123  # Это можно получить как результат работы нейросети или вручную
visualize_tarot_sphere(num_points=num_tractovki, embedding_dim=embedding_dim, active_index=active_index, radius=radius, card_name=target_card)

# Получение выводов сети для всех карт
input_contexts = {name: emb for name, emb in all_embeddings.items()}
outputs = tarot_network(input_contexts)

# Вывод трактовки для каждой карты
for card_name, (best_tractovka, idx) in outputs.items():
    print(f"Трактовка для карты {card_name}: {best_tractovka}, индекс: {idx}")

# Пример визуализации активной трактовки на сфере
def visualize_active_tractovka(card_name, active_index):
    # Визуализация активной трактовки для заданной карты
    print(f"Визуализируем активную трактовку для карты {card_name}, индекс: {active_index}")
    visualize_tarot_sphere(num_points=num_tractovki, embedding_dim=embedding_dim, active_index=active_index, radius=radius, card_name=card_name)

# Визуализируем активную трактовку для карты "The Fool"
visualize_active_tractovka("The Fool", active_index)

