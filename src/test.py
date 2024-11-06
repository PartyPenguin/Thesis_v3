import numpy as np

def positional_embedding(coords, L=10):
    """
    Generate Fourier-based positional embedding for 3D coordinates.

    Parameters:
    - coords: np.array of shape (N, 3) where N is the number of points, each with (x, y, z) coordinates.
    - L: int, the number of frequency bands.

    Returns:
    - np.array of shape (N, 6 * L) containing the Fourier embeddings for each point.
    """
    coords = np.asarray(coords)  # Ensure input is a numpy array
    embed_list = [coords]  # Start with the original coordinates
    
    for i in range(L):
        freq = 2.0 ** i  # Frequency increases as 2^i
        embed_list.append(np.sin(freq * np.pi * coords))  # Sine component
        embed_list.append(np.cos(freq * np.pi * coords))  # Cosine component

    # Concatenate all embeddings along the last axis
    return np.concatenate(embed_list, axis=-1)

# Example usage:
# Assuming you have a set of 3D coordinates
coords = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Shape (N, 3), N=2 in this example
L = 10  # Number of frequency bands
embedded_coords = positional_embedding(coords, L)
print("Positional Embeddings Shape:", embedded_coords.shape)
print(embedded_coords)