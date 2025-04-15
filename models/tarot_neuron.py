# tarot_neuron.py — структура нейрона и карты с 3D-сферой трактовок

import torch
import torch.nn as nn
import torch.nn.functional as F

class TarotCardNeuron(nn.Module):
    def __init__(self, name: str, embedding_dim: int = 64, num_tractovki: int = 500, radius: float = 5.0):
        super().__init__()
        self.name = name
        self.embedding_dim = embedding_dim
        self.num_tractovki = num_tractovki
        self.radius = radius

        # Центральная точка карты (ядро смысла)
        self.center = nn.Parameter(torch.zeros(embedding_dim))

        # Трактовки на сфере вокруг центра
        self.tractovki = nn.Parameter(F.normalize(self._init_tractovki(), dim=-1) * radius)

    def _init_tractovki(self):
        indices = torch.arange(0, self.num_tractovki, dtype=torch.float) + 0.5
        phi = torch.acos(1 - 2 * indices / self.num_tractovki)
        theta = torch.pi * (1 + 5**0.5) * indices

        x = self.radius * torch.sin(phi) * torch.cos(theta)
        y = self.radius * torch.sin(phi) * torch.sin(theta)
        z = self.radius * torch.cos(phi)

        # Вставляем координаты в начало вектора, остальные обнуляем
        points = torch.zeros(self.num_tractovki, self.embedding_dim)
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        return points

    def forward(self, context_vector: torch.Tensor):
        # Находим ближайшую трактовку к текущему контексту
        # Контекст проецируем относительно центра
        direction = F.normalize(context_vector - self.center, dim=-1)
        projections = torch.matmul(self.tractovki, direction)
        best_index = torch.argmax(projections)
        return self.tractovki[best_index], best_index


class TarotNeuronNetwork(nn.Module):
    def __init__(self, card_names: list[str], embedding_dim: int = 64, num_tractovki: int = 500):
        super().__init__()
        self.cards = nn.ModuleDict({
            name: TarotCardNeuron(name, embedding_dim, num_tractovki)
            for name in card_names
        })

    def forward(self, input_contexts: dict[str, torch.Tensor]):
        # input_contexts: {card_name: context_vector}
        outputs = {}
        for name, context in input_contexts.items():
            if name in self.cards:
                best_tractovka, idx = self.cards[name](context)
                outputs[name] = (best_tractovka, idx)
        return outputs