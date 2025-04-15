# Enhanced decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedTarotInterpreter(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Tractovka embedding processor
        self.tractovka_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Context circle embedding processor
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Card embedding processor
        self.card_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Transformer-based decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model = hidden_dim,
            nhead = 8,
            dim_feedforward = hidden_dim * 4,
            batch_first = True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers = 3
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Token embeddings (would be shared with an encoder in a real implementation)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(1024, hidden_dim)  # Position embeddings up to 1024 tokens

    def forward(self,
                tractovka_vector: torch.Tensor,
                context_vector: torch.Tensor = None,
                card_vector: torch.Tensor = None,
                prev_tokens: torch.Tensor = None,
                max_len: int = 100):
        """
        Generate interpretation text from tractovka and context

        Args:
            tractovka_vector: Vector of the selected tractovka point
            context_vector: Vector of the context circle (optional)
            card_vector: Vector of the card (optional)
            prev_tokens: Previously generated tokens for autoregressive generation
            max_len: Maximum length of generated sequence

        Returns:
            Token logits for next tokens
        """
        batch_size = tractovka_vector.size(0)

        # Encode inputs
        tractovka_enc = self.tractovka_encoder(tractovka_vector)

        if context_vector is None:
            context_enc = torch.zeros_like(tractovka_enc)
        else:
            context_enc = self.context_encoder(context_vector)

        if card_vector is None:
            card_enc = torch.zeros_like(tractovka_enc)
        else:
            card_enc = self.card_encoder(card_vector)

        # Fuse all information
        memory = self.fusion(torch.cat([tractovka_enc, context_enc, card_enc], dim = -1))
        memory = memory.unsqueeze(1)  # (B, 1, H)

        # For inference (autoregressive generation)
        if prev_tokens is None:
            # Start with just a BOS token (0)
            tgt = torch.zeros(batch_size, 1, dtype = torch.long, device = tractovka_vector.device)

            outputs = []

            for i in range(max_len):
                # Get embeddings with positions
                tgt_emb = self.token_embedding(tgt)
                pos_ids = torch.arange(tgt.size(1), device = tgt.device).unsqueeze(0).expand_as(tgt)
                pos_emb = self.pos_embedding(pos_ids)
                tgt_emb = tgt_emb + pos_emb

                # Create causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

                # Decode
                out = self.transformer_decoder(
                    tgt_emb, memory, tgt_mask = tgt_mask
                )

                # Get next token logits
                logits = self.output(out[:, -1:])
                outputs.append(logits)

                # Predict next token
                next_token = torch.argmax(logits, dim = -1)
                tgt = torch.cat([tgt, next_token], dim = 1)

                # Stop if EOS token (1) is generated
                if next_token.item() == 1:
                    break

            return torch.cat(outputs, dim = 1)

        # For training (teacher forcing)
        else:
            # Embed target tokens
            tgt_emb = self.token_embedding(prev_tokens)
            pos_ids = torch.arange(prev_tokens.size(1), device = prev_tokens.device).unsqueeze(0).expand_as(prev_tokens)
            pos_emb = self.pos_embedding(pos_ids)
            tgt_emb = tgt_emb + pos_emb

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(prev_tokens.size(1)).to(prev_tokens.device)

            # Decode
            out = self.transformer_decoder(
                tgt_emb, memory, tgt_mask = tgt_mask
            )

            # Get logits for all positions
            logits = self.output(out)
            return logits

    def generate_text(self, tractovka_vector, context_vector = None, card_vector = None,
                      tokenizer = None, max_len = 100):
        """Generate text from vectors using greedy decoding"""
        logits = self.forward(
            tractovka_vector.unsqueeze(0),
            None if context_vector is None else context_vector.unsqueeze(0),
            None if card_vector is None else card_vector.unsqueeze(0),
            max_len = max_len
        )

        token_ids = torch.argmax(logits, dim = -1).squeeze(0).tolist()

        if tokenizer:
            return tokenizer.decode(token_ids)
        else:
            return token_ids