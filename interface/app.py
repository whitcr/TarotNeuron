# interface/app.py
import streamlit as st
import torch
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.full_system import TarotSystem
from utils.visualizer import visualize_tarot_sphere, visualize_card_circle

# Load card names
with open('data/card_names.json', 'r') as f:
    card_names_dict = json.load(f)
    CARDS = [(int(k), v) for k, v in card_names_dict.items()]


# Initialize model
@st.cache_resource
def load_model():
    model = TarotSystem(embedding_dim = 64, num_tractovki = 360)
    return model


model = load_model()

# UI
st.title("Tarot Neural Interpreter")

# Select cards
st.subheader("Select Cards for Reading")
selected_cards = st.multiselect(
    "Choose cards:",
    options = CARDS,
    format_func = lambda x: f"{x[0]} - {x[1]}",
    default = [(0, "The Fool")]
)

# Reading context
context = st.text_area("Enter your question or context:", "What guidance do I need right now?")

# Generate interpretation
if st.button("Generate Interpretation"):
    if selected_cards:
        card_indices = [card[0] for card in selected_cards]

        with st.spinner("Generating interpretation..."):
            # Forward pass
            output = model.forward(card_indices, context)

            # Show results
            st.subheader("Interpretation")
            st.write("(Placeholder for actual text generation)")

            # Visualize cards
            st.subheader("Card Visualization")
            selected_card = st.selectbox(
                "Select card to visualize:",
                options = [card[1] for card in selected_cards]
            )

            # Get card index and show visualization
            card_idx = next(i for i, (_, name) in enumerate(selected_cards) if name == selected_card)
            st.pyplot(
                visualize_tarot_sphere(
                    num_points = model.neuron_network.cards[selected_card].num_tractovki,
                    active_index = 123,  # Would be actual index from model
                    card_name = selected_card,
                    show = False
                )
            )