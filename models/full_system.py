# models/improved_full_system.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tarot_neuron import ImprovedTarotNeuronNetwork
from models.context_selector import EnhancedContextSelector
from models.decoder import EnhancedTarotInterpreter


class ImprovedTarotSystem(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256,
                 num_contexts: int = 12, points_per_context: int = 30,
                 vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Load card names
        with open('data/card_names.json', 'r') as f:
            card_names_dict = json.load(f)
            self.card_names = [card_names_dict[str(i)] for i in range(len(card_names_dict))]
            self.card_indices = {name: idx for idx, name in enumerate(self.card_names)}

        # Initialize card embeddings (learnable)
        self.card_embeddings = nn.Parameter(torch.randn(len(self.card_names), embedding_dim))

        # Initialize components
        self.context_selector = EnhancedContextSelector(
            self.card_names, embedding_dim, num_contexts
        )

        self.neuron_network = ImprovedTarotNeuronNetwork(
            self.card_names, embedding_dim, num_contexts, points_per_context
        )

        self.decoder = EnhancedTarotInterpreter(
            embedding_dim, hidden_dim, vocab_size
        )

        # Question encoder
        self.question_encoder = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = embedding_dim,
            num_layers = 2,
            batch_first = True,
            bidirectional = True
        )
        self.question_projection = nn.Linear(embedding_dim * 2, embedding_dim)

        # Learnable token embeddings for question encoding
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def encode_question(self, question_tokens):
        """
        Encode a question into a context vector

        Args:
            question_tokens: Tokenized question (batch_size, seq_len)

        Returns:
            Question embedding vector
        """
        token_embeds = self.token_embeddings(question_tokens)  # (B, L, D)
        _, (hidden, _) = self.question_encoder(token_embeds)

        # Combine bidirectional states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim = -1)  # (B, 2*D)
        question_emb = self.question_projection(hidden)  # (B, D)

        return question_emb

    def forward(self, card_indices: list[int], question_tokens = None):
        """
        Generate tarot reading interpretation

        Args:
            card_indices: List of card indices in the reading
            question_tokens: Tokenized question (optional)

        Returns:
            Interpretation token logits and context info
        """
        # Get card embeddings for selected cards
        selected_cards = [self.card_names[i] for i in card_indices]
        input_contexts = {
            name: self.card_embeddings[i]
            for i, name in zip(card_indices, selected_cards)
        }

        # Process question if provided
        question_embedding = None
        if question_tokens is not None:
            question_embedding = self.encode_question(question_tokens)

        # Get context vectors using enhanced context selector
        context_vectors, context_prefs = self.context_selector.select_context(
            input_contexts, question_embedding
        )

        # Get tractovki for each card
        tractovki_results = self.neuron_network(context_vectors)

        # Extract vectors, indices, and context indices
        tractovka_vectors = {}
        tractovka_indices = {}
        context_indices = {}

        for name, (vector, idx, context_idx) in tractovki_results.items():
            tractovka_vectors[name] = vector
            tractovka_indices[name] = idx
            context_indices[name] = context_idx

        # Combine tractovki for all cards in the reading
        combined_tractovka = torch.stack(list(tractovka_vectors.values())).mean(dim = 0)

        # Get card vectors
        card_vectors = torch.stack([self.card_embeddings[i] for i in card_indices])
        combined_card = card_vectors.mean(dim = 0)

        # Get context vectors for the activated tractovki
        context_vecs = []
        for name, context_idx in context_indices.items():
            # Get the context embedding for this card's activated context
            card_neuron = self.neuron_network.cards[name]
            context_meanings = card_neuron.context_meanings
            context_vecs.append(context_meanings[context_idx])

        combined_context = torch.stack(context_vecs).mean(dim = 0) if context_vecs else None

        # Generate interpretation using enhanced decoder
        logits = self.decoder(
            tractovka_vector = combined_tractovka.unsqueeze(0),
            context_vector = None if combined_context is None else combined_context.unsqueeze(0),
            card_vector = combined_card.unsqueeze(0)
        )

        # Return everything needed for analysis
        return {
            'logits': logits,
            'tractovka_indices': tractovka_indices,
            'context_indices': context_indices,
            'context_preferences': context_prefs
        }

    def generate_interpretation(self, card_indices, question = None, tokenizer = None, max_len = 100):
        """
        Generate a human-readable interpretation for a tarot reading

        Args:
            card_indices: List of card indices in the reading
            question: Question text (optional)
            tokenizer: Tokenizer for encoding/decoding text
            max_len: Maximum length of generated text

        Returns:
            Generated interpretation text and analysis info
        """
        # Prepare input
        question_tokens = None
        if question and tokenizer:
            question_tokens = torch.tensor([tokenizer.encode(question)], dtype = torch.long)

        # Get model outputs
        with torch.no_grad():
            outputs = self.forward(card_indices, question_tokens)

        # Generate text from logits
        token_ids = torch.argmax(outputs['logits'], dim = -1).squeeze(0).tolist()

        # Decode to text
        if tokenizer:
            interpretation = tokenizer.decode(token_ids)
        else:
            interpretation = f"[Token IDs: {token_ids[:10]}...]"

        # Format results with analysis
        result = {
            'interpretation': interpretation,
            'cards': [self.card_names[i] for i in card_indices],
            'question': question,
            'analysis': {
                'tractovka_indices': {self.card_names[i]: idx.item()
                                      for i, idx in zip(card_indices, outputs['tractovka_indices'].values())},
                'context_indices': {self.card_names[i]: idx.item()
                                    for i, idx in zip(card_indices, outputs['context_indices'].values())}
            }
        }

        return result