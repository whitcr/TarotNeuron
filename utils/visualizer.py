# Enhanced visualizer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def visualize_context_circles(card_neuron, active_index = None, context_weights = None, show = True):
    """
    Visualize the improved tarot card neuron with highlighted context circles

    Args:
        card_neuron: The ImprovedTarotCardNeuron instance
        active_index: Index of active tractovka (optional)
        context_weights: Weights of each context from context selector (optional)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111, projection = '3d')

    # Get tractovki coordinates and convert to numpy
    tractovki = card_neuron.tractovki.detach().cpu().numpy()
    x = tractovki[:, 0]
    y = tractovki[:, 1]
    z = tractovki[:, 2]

    # Get center coordinates
    center = card_neuron.center.detach().cpu().numpy()

    # Prepare colors - one distinct color per context
    colors = list(mcolors.TABLEAU_COLORS.values())
    num_contexts = card_neuron.num_contexts
    points_per_context = card_neuron.points_per_context

    # Plot each context circle
    for c in range(num_contexts):
        start_idx = c * points_per_context
        end_idx = start_idx + points_per_context

        # Get points for this context
        cx = x[start_idx:end_idx]
        cy = y[start_idx:end_idx]
        cz = z[start_idx:end_idx]

        # Set circle properties based on context weights
        circle_alpha = 0.7
        circle_size = 30
        circle_width = 1

        if context_weights is not None:
            # Make more relevant contexts more prominent
            weight = context_weights[c].item()
            circle_alpha = 0.3 + weight * 0.7  # Scale alpha by weight
            circle_size = 20 + weight * 60  # Scale size by weight
            circle_width = 1 + weight * 3  # Scale line width by weight

        color = colors[c % len(colors)]

        # Plot this context's points
        ax.scatter(
            cx, cy, cz,
            label = f"Context {c}" if c < 8 else None,
            color = color, alpha = circle_alpha, s = circle_size
            )

        # Draw a line connecting the points in the circle
        circle_x = np.append(cx, cx[0])
        circle_y = np.append(cy, cy[0])
        circle_z = np.append(cz, cz[0])
        ax.plot(
            circle_x, circle_y, circle_z, color = color, alpha = circle_alpha,
            linewidth = circle_width
            )

    # Plot the center point (card)
    ax.scatter(
        [center[0]], [center[1]], [center[2]], color = 'black', s = 150,
        marker = '*', label = f"Card: {card_neuron.name}"
        )

    # Highlight active tractovka if provided
    if active_index is not None:
        ax.scatter(
            [x[active_index]], [y[active_index]], [z[active_index]],
            color = 'red', s = 120, marker = 'o', label = 'Active Tractovka'
            )

        # Draw line from center to active tractovka
        ax.plot(
            [center[0], x[active_index]],
            [center[1], y[active_index]],
            [center[2], z[active_index]],
            'r--', linewidth = 3
            )

        # Get context of active tractovka
        context_idx = active_index // points_per_context
        ax.text(
            x[active_index], y[active_index], z[active_index],
            f"Context {context_idx}", color = 'red', fontsize = 12
            )

    ax.set_title(f"Context Circles: {card_neuron.name}", fontsize = 16)
    ax.legend(loc = 'upper right', bbox_to_anchor = (1.3, 1))

    # Add grid and improve axes
    ax.grid(True, alpha = 0.3)
    ax.set_xlabel('X', fontsize = 12)
    ax.set_ylabel('Y', fontsize = 12)
    ax.set_zlabel('Z', fontsize = 12)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_multi_card_reading(neuron_network, card_names, tractovka_indices, context_indices):
    """
    Visualize active tractovki across multiple cards in a reading

    Args:
        neuron_network: The ImprovedTarotNeuronNetwork instance
        card_names: List of card names in the reading
        tractovka_indices: Dictionary mapping card names to active tractovka indices
        context_indices: Dictionary mapping card names to active context indices
    """
    num_cards = len(card_names)

    # Create grid layout based on number of cards
    cols = min(3, num_cards)
    rows = (num_cards + cols - 1) // cols

    fig = plt.figure(figsize = (6 * cols, 5 * rows))

    for i, card_name in enumerate(card_names):
        if card_name not in neuron_network.cards:
            continue

        card_neuron = neuron_network.cards[card_name]
        active_index = tractovka_indices.get(card_name, None)

        # Create 3D subplot
        ax = fig.add_subplot(rows, cols, i + 1, projection = '3d')

        # Get tractovki coordinates
        tractovki = card_neuron.tractovki.detach().cpu().numpy()
        x = tractovki[:, 0]
        y = tractovki[:, 1]
        z = tractovki[:, 2]

        # Get center coordinates
        center = card_neuron.center.detach().cpu().numpy()

        # Prepare colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        points_per_context = card_neuron.points_per_context

        # Plot each context circle
        for c in range(card_neuron.num_contexts):
            start_idx = c * points_per_context
            end_idx = start_idx + points_per_context

            # Get points for this context
            cx = x[start_idx:end_idx]
            cy = y[start_idx:end_idx]
            cz = z[start_idx:end_idx]

            # Highlight active context
            context_idx = context_indices.get(card_name, -1)
            alpha = 0.9 if c == context_idx else 0.3
            size = 40 if c == context_idx else 20
            color = colors[c % len(colors)]

            # Plot points in this context
            ax.scatter(cx, cy, cz, color = color, alpha = alpha, s = size)

            # Only draw connecting line for the active context
            if c == context_idx:
                circle_x = np.append(cx, cx[0])
                circle_y = np.append(cy, cy[0])
                circle_z = np.append(cz, cz[0])
                ax.plot(circle_x, circle_y, circle_z, color = color, linewidth = 2)

        # Plot the center (card)
        ax.scatter([center[0]], [center[1]], [center[2]], color = 'black', s = 100, marker = '*')

        # Highlight active tractovka if available
        if active_index is not None:
            ax.scatter(
                [x[active_index]], [y[active_index]], [z[active_index]],
                color = 'red', s = 100
                )

            # Draw line from center to active tractovka
            ax.plot(
                [center[0], x[active_index]],
                [center[1], y[active_index]],
                [center[2], z[active_index]],
                'r--', linewidth = 2
                )

        ax.set_title(f"{card_name}", fontsize = 14)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)

    # Add main title
    fig.suptitle("Tarot Reading Visualization", fontsize = 20, y = 0.98)

    return fig