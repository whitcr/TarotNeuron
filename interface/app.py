# Updated interface/app.py
import streamlit as st
import torch
import json
import numpy as np
import sys
import os

from models.full_system import ImprovedTarotSystem
from utils.visualizer import visualize_context_circles, visualize_multi_card_reading
from utils.tokenizer import SimpleTokenizer
# Load card names
with open('data/card_names.json', 'r') as f:
    card_names_dict = json.load(f)
    CARDS = [(int(k), v) for k, v in card_names_dict.items()]
# Load tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer = SimpleTokenizer()
    tokenizer.load('data/vocabulary.pkl')
    return tokenizer
# Initialize model
@st.cache_resource
def load_model():
    model = ImprovedTarotSystem(
        embedding_dim=64,
        hidden_dim=256,
        num_contexts=12,
        points_per_context=30
    )
    # Load pre-trained weights if available
    try:
        model.load_state_dict(torch.load('checkpoints/tarot_model.pt', map_location='cpu'))
    except:
        st.warning("No pre-trained model found. Using initialized weights.")

    return model
# UI Components
def render_sidebar():
    st.sidebar.title("TarotNeuron Options")

    # Analysis options
    st.sidebar.subheader("Analysis Mode")
    show_context_distribution = st.sidebar.checkbox("Show Context Distribution", value=True)
    show_detailed_visualization = st.sidebar.checkbox("Show Detailed Visualization", value=False)

    # Advanced options
    st.sidebar.subheader("Advanced Settings")
    model_params = {
        'max_length': st.sidebar.slider("Max Interpretation Length", 50, 300, 150),
        'temperature': st.sidebar.slider("Interpretation Creativity", 0.1, 1.5, 0.7),
    }

    return {
        'show_context_distribution': show_context_distribution,
        'show_detailed_visualization': show_detailed_visualization,
        'model_params': model_params
    }
# Main app
def main():
    st.title("🔮 Neural Tarot Interpreter")
    st.subheader("AI-powered tarot reading with contextual understanding")

    # Load resources
    model = load_model()
    tokenizer = load_tokenizer()

    # Get sidebar options
    options = render_sidebar()

    # Main section
    col1, col2 = st.columns([2, 1])

    with col1:
        # Reading context
        context = st.text_area(
            "Enter your question or situation",
            "What does the universe want me to know right now?",
            height = 100
            )

        # Select reading type
        reading_type = st.selectbox(
            "Select a reading spread",
            ["Single Card", "Three Card Spread", "Celtic Cross"]
        )

        # Draw cards button
        if st.button("Draw Cards 🎴"):
            with st.spinner("Connecting with the arcana..."):
                # Process input context
                context_tokens = tokenizer.encode(context)
                context_tensor = torch.tensor(context_tokens).unsqueeze(0)

                # Get number of cards based on reading type
                num_cards = 1 if reading_type == "Single Card" else 3 if reading_type == "Three Card Spread" else 10

                # Generate card predictions
                with torch.no_grad():
                    card_indices, interpretation, context_weights = model.generate_reading(
                        context_tensor,
                        num_cards = num_cards,
                        max_length = options['model_params']['max_length'],
                        temperature = options['model_params']['temperature']
                    )

                # Store results in session state
                st.session_state.cards = [CARDS[idx] for idx in card_indices]
                st.session_state.interpretation = interpretation
                st.session_state.context_weights = context_weights
                st.session_state.reading_type = reading_type
                st.session_state.context = context

    with col2:
        # Information about the app
        st.info(
            """
            **How It Works**

            This neural tarot reader analyzes your question across 12 contextual 
            dimensions to deliver personalized insights. The model has been trained 
            on thousands of tarot readings to understand the complex relationships 
            between cards, questions, and interpretations.
            """
            )

        # Card meanings reference
        with st.expander("Card Reference Guide"):
            st.write("Hover over any card in your reading to see its basic meaning.")
            # This could be expanded with a searchable list of cards and meanings

    # Display results if available
    if 'cards' in st.session_state:
        st.header("Your Reading")

        # Display cards and visualization
        col_cards, col_viz = st.columns([3, 2])

        with col_cards:
            # Show cards based on reading type
            if st.session_state.reading_type == "Single Card":
                card_id, card_name = st.session_state.cards[0]
                st.image(f"assets/cards/{card_id}.jpg", width = 200, caption = card_name)
            else:
                # Visualize multi-card reading
                fig = visualize_multi_card_reading(
                    st.session_state.cards,
                    spread_type = st.session_state.reading_type
                )
                st.pyplot(fig)

                # List all cards in the reading
                st.subheader("Cards in Your Reading")
                for i, (card_id, card_name) in enumerate(st.session_state.cards):
                    position_name = ""
                    if st.session_state.reading_type == "Three Card Spread":
                        position_name = ["Past", "Present", "Future"][i]
                    elif st.session_state.reading_type == "Celtic Cross":
                        positions = ["Present", "Challenge", "Foundation", "Recent Past",
                                     "Potential", "Near Future", "Self", "Environment",
                                     "Hopes/Fears", "Outcome"]
                        position_name = positions[i]

                    st.write(f"**{position_name}**: {card_name}")

        with col_viz:
            # Show context distribution if enabled
            if options['show_context_distribution']:
                st.subheader("Question Analysis")
                context_fig = visualize_context_circles(st.session_state.context_weights)
                st.pyplot(context_fig)

                # Add explanation
                st.caption(
                    """
                    This visualization shows how your question was analyzed across 
                    12 contextual dimensions, including relationships, career, 
                    spiritual growth, and more.
                    """
                    )

        # Show interpretation
        st.subheader("Interpretation")
        st.write(st.session_state.interpretation)

        # Detailed visualization section
        if options['show_detailed_visualization'] and st.session_state.reading_type != "Single Card":
            st.subheader("Card Relationship Analysis")
            st.warning("Advanced visualization coming soon...")

        # Add options to save or share reading
        col_save, col_share = st.columns(2)
        with col_save:
            st.download_button(
                "Save Reading",
                data = json.dumps(
                    {
                        "context": st.session_state.context,
                        "reading_type": st.session_state.reading_type,
                        "cards": [(id, name) for id, name in st.session_state.cards],
                        "interpretation": st.session_state.interpretation,
                        "timestamp": str(datetime.datetime.now())
                    }
                ),
                file_name = "tarot_reading.json",
                mime = "application/json"
            )
        with col_share:
            st.button("Share Reading", disabled = True, help = "Coming soon!")

        # Offer a new reading
        if st.button("Start a New Reading"):
            for key in ['cards', 'interpretation', 'context_weights', 'reading_type']:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()


if __name__ == "__main__":
    # Import datetime only if main is executed
    import datetime

    # Set page configuration
    st.set_page_config(
        page_title = "Neural Tarot Interpreter",
        page_icon = "🔮",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    # Run the app
    main()