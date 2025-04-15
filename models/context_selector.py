# Enhanced context_selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedContextSelector(nn.Module):
    def __init__(self, card_names: list[str], embedding_dim: int = 64, num_contexts: int = 12):
        super(EnhancedContextSelector, self).__init__()
        self.card_names = card_names
        self.embedding_dim = embedding_dim
        self.num_contexts = num_contexts

        # Standard attention components
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Context-aware components
        self.context_embeddings = nn.Parameter(torch.randn(num_contexts, embedding_dim))
        self.context_attention = nn.Linear(embedding_dim * 2, num_contexts)

        # Question understanding
        self.question_encoder = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = embedding_dim,
            num_layers = 1,
            batch_first = True
        )

        # Final projection
        self.output_proj = nn.Linear(embedding_dim * 2, embedding_dim)

    def encode_question(self, question_tokens):
        """Encode the question text into a context vector"""
        # In a real implementation, this would use proper tokenization
        # Here we assume question_tokens is already a tensor of token embeddings
        _, (hidden, _) = self.question_encoder(question_tokens)
        return hidden.squeeze(0)

    def forward(self, target_card: str, all_embeddings: dict[str, torch.Tensor],
                question_embedding: torch.Tensor = None):
        """
        Select context for target_card based on other cards and question

        Args:
            target_card: Name of the card to find interpretation for
            all_embeddings: Dictionary of card embeddings
            question_embedding: Embedding of the user's question (optional)

        Returns:
            Context vector with context preference weights
        """
        # Get target card embedding
        queries = self.query_proj(all_embeddings[target_card].unsqueeze(0))  # (1, D)

        # Process other cards
        keys = []
        values = []
        for name, emb in all_embeddings.items():
            if name != target_card:
                keys.append(self.key_proj(emb))
                values.append(self.value_proj(emb))

        if keys:  # If there are other cards
            keys = torch.stack(keys)  # (N-1, D)
            values = torch.stack(values)  # (N-1, D)

            # Card attention
            attn_scores = torch.matmul(queries, keys.T) / (self.embedding_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim = -1)
            card_context = torch.matmul(attn_weights, values).squeeze(0)  # (D)
        else:
            # If only one card, use its embedding as context
            card_context = all_embeddings[target_card]

        # Combine with question if provided
        if question_embedding is not None:
            combined = torch.cat([card_context, question_embedding], dim = -1)
        else:
            # Duplicate the card context if no question
            combined = torch.cat([card_context, card_context], dim = -1)

        # Calculate context circle preferences based on combined info
        context_weights = F.softmax(self.context_attention(combined), dim = -1)  # (num_contexts)

        # Weight the contexts
        weighted_context_embs = torch.matmul(context_weights, self.context_embeddings)  # (D)

        # Combine card context with context preferences
        final_context = self.output_proj(torch.cat([card_context, weighted_context_embs], dim = -1))

        return final_context, context_weights

    def select_context(self, input_contexts: dict[str, torch.Tensor], question: str = None):
        """Process all cards in the reading to generate context vectors"""
        # Encode question if provided
        question_embedding = None
        if question:
            # In a real implementation, this would tokenize the question
            # For now, we'll just create a dummy embedding
            question_embedding = torch.randn(self.embedding_dim)

        context_vectors = {}
        context_preferences = {}
        for target_card in input_contexts:
            vector, prefs = self.forward(target_card, input_contexts, question_embedding)
            context_vectors[target_card] = vector
            context_preferences[target_card] = prefs

        return context_vectors, context_preferences