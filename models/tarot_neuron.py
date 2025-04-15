# Improved tarot_neuron.py with context circles and interpretation points

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedTarotCardNeuron(nn.Module):
    def __init__(self, name: str, embedding_dim: int = 64, num_contexts: int = 12,
                 points_per_context: int = 30, radius: float = 5.0):
        super().__init__()
        self.name = name
        self.embedding_dim = embedding_dim
        self.num_contexts = num_contexts
        self.points_per_context = points_per_context
        self.num_tractovki = num_contexts * points_per_context
        self.radius = radius

        # Center point of the card (core meaning)
        self.center = nn.Parameter(torch.zeros(embedding_dim))

        # Context circles and their interpretation points
        self.tractovki = nn.Parameter(self._init_contextual_tractovki())

        # Each context has a semantic meaning (learnable)
        self.context_meanings = nn.Parameter(torch.randn(num_contexts, embedding_dim))

    def _init_contextual_tractovki(self):
        """Initialize tractovki arranged in context circles"""
        tractovki = torch.zeros(self.num_tractovki, self.embedding_dim)

        # For each context, create a circle of interpretation points
        for c in range(self.num_contexts):
            # Calculate the position of this context circle
            # We use spherical coordinates to position each context circle
            # Each context sits at a different latitude on the sphere
            context_phi = torch.tensor(math.pi * (c + 0.5) / self.num_contexts)

            # For each interpretation point in this context
            for p in range(self.points_per_context):
                idx = c * self.points_per_context + p

                # Position this point on its context circle (different longitudes)
                theta = torch.tensor(2 * math.pi * p / self.points_per_context)

                # Convert spherical coordinates to 3D
                x = self.radius * torch.sin(context_phi) * torch.cos(theta)
                y = self.radius * torch.sin(context_phi) * torch.sin(theta)
                z = self.radius * torch.cos(context_phi)

                # First 3 dimensions are spatial coordinates
                tractovki[idx, 0] = x
                tractovki[idx, 1] = y
                tractovki[idx, 2] = z

                # Initialize remaining dimensions with small random values
                if self.embedding_dim > 3:
                    tractovki[idx, 3:] = torch.randn(self.embedding_dim - 3) * 0.1

        return tractovki

    def get_context_index(self, tractovka_index):
        """Convert tractovka index to context index"""
        return tractovka_index // self.points_per_context

    def forward(self, context_vector: torch.Tensor):
        """
        Find the most relevant interpretation given a context vector
        Returns the tractovka vector, its index, and the context circle index
        """
        # Find direction from center to context
        direction = F.normalize(context_vector - self.center, dim = -1)

        # Calculate similarity between context direction and all tractovki
        similarities = torch.matmul(F.normalize(self.tractovki, dim = -1), direction)

        # Find best tractovka
        best_index = torch.argmax(similarities)
        context_index = self.get_context_index(best_index)

        return self.tractovki[best_index], best_index, context_index

    def get_context_points(self, context_index):
        """Get all tractovki belonging to a specific context circle"""
        start_idx = context_index * self.points_per_context
        end_idx = start_idx + self.points_per_context
        return self.tractovki[start_idx:end_idx]


class ImprovedTarotNeuronNetwork(nn.Module):
    def __init__(self, card_names: list[str], embedding_dim: int = 64,
                 num_contexts: int = 12, points_per_context: int = 30):
        super().__init__()
        self.cards = nn.ModuleDict(
            {
                name: ImprovedTarotCardNeuron(
                    name, embedding_dim, num_contexts, points_per_context
                ) for name in card_names
            }
        )

    def forward(self, input_contexts: dict[str, torch.Tensor]):
        """
        Forward pass for the entire neural network
        Returns more detailed information about interpretations
        """
        outputs = {}
        for name, context in input_contexts.items():
            if name in self.cards:
                best_tractovka, idx, context_idx = self.cards[name](context)
                outputs[name] = (best_tractovka, idx, context_idx)
        return outputs