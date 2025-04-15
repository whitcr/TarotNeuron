import json

import torch
from torch import nn

from models.context_selector import ContextSelector
from models.tarot_neuron import TarotNeuronNetwork
from models.decoder import TarotInterpreter


class TarotSystem:
    def __init__(self, embedding_dim: int = 64, num_tractovki: int = 500):
        self.embedding_dim = embedding_dim

        # Load card names
        with open('data/card_names.json', 'r') as f:
            card_names_dict = json.load(f)
            self.card_names = [card_names_dict[str(i)] for i in range(len(card_names_dict))]

        # Initialize components
        self.context_selector = ContextSelector(self.card_names, embedding_dim)
        self.neuron_network = TarotNeuronNetwork(self.card_names, embedding_dim, num_tractovki)
        self.decoder = TarotInterpreter(embedding_dim = embedding_dim)

        # Card embeddings (would be trained/loaded in practice)
        self.card_embeddings = nn.Parameter(torch.randn(len(self.card_names), embedding_dim))

    def forward(self, card_indices: list[int], context_text: str = None):
        """
        Generate interpretation for a Tarot reading

        Args:
            card_indices: List of card indices in the reading
            context_text: Optional context text for the reading

        Returns:
            Logits for interpretation tokens
        """
        # Create input contexts dictionary
        selected_cards = [self.card_names[i] for i in card_indices]
        input_contexts = {
            name: self.card_embeddings[i]
            for i, name in zip(card_indices, selected_cards)
        }

        # Get context vectors using attention
        context_vectors = self.context_selector.select_context(input_contexts)

        # Get tractovki for each card
        tractovki_results = self.neuron_network(context_vectors)

        # Combine tractovki for all cards in the reading
        combined_tractovka = torch.stack([result[0] for result in tractovki_results.values()]).mean(dim = 0)

        # Decode to text tokens (just logits for now)
        return self.decoder(combined_tractovka.unsqueeze(0))