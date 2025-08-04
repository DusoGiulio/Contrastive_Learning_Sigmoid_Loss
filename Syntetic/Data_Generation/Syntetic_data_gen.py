# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:41:04 2025

@author: lport
"""

import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os

def generate_correlated_embeddings(
    num_samples: int,
    num_classes: int,
    text_embedding_dim: int,
    image_embedding_dim: int,
    latent_dim: int = 64,
    noise_scale: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates correlated text and image embeddings organized by class.

    This function creates two sets of embeddings (t_emb, i_emb) with the
    following properties:
    1.  A shared semantic structure: If two text embeddings from different
        classes are similar, their corresponding image embeddings will also be
        similar.
    2.  Separate vector spaces: Text and image embeddings have different,
        independent dimensions.
    3.  Class association: Each (text, image) pair belongs to one of C classes.

    Args:
        num_samples (int): The total number of embedding pairs to generate.
        num_classes (int): The number of distinct classes to model.
        text_embedding_dim (int): The dimensionality of the text embeddings.
        image_embedding_dim (int): The dimensionality of the image embeddings.
        latent_dim (int): The dimensionality of the shared latent space where
                          class concepts are defined.
        noise_scale (float): The standard deviation of the Gaussian noise added
                             to each embedding to create intra-class variance.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - t_emb (np.ndarray): The matrix of text embeddings (shape: num_samples x text_embedding_dim).
            - i_emb (np.ndarray): The matrix of image embeddings (shape: num_samples x image_embedding_dim).
            - labels (np.ndarray): An array of class labels for each pair (shape: num_samples).
    """
    print(f"Generating {num_samples} samples across {num_classes} classes...")
    
    # 1. Create class prototypes in a shared latent space
    # These vectors represent the "pure concept" of each class.
    np.random.seed(42) # for reproducibility
    latent_prototypes = np.random.randn(num_classes, latent_dim)
    # Normalize prototypes to have unit length for consistent similarity
    latent_prototypes /= np.linalg.norm(latent_prototypes, axis=1, keepdims=True)

    # 2. Create separate projection matrices for text and image spaces
    # These matrices will map the shared concepts into their respective, 
    #different vector spaces.
    projection_to_text = np.random.randn(latent_dim, text_embedding_dim)
    projection_to_image = np.random.randn(latent_dim, image_embedding_dim)

    # 3. Project latent prototypes to create class centroids in each space
    text_class_centroids = latent_prototypes @ projection_to_text
    image_class_centroids = latent_prototypes @ projection_to_image

    # 4. Generate individual samples for each class
    t_emb_list =[]
    i_emb_list =[]
    labels_list =[]

    # Assign samples to classes
    sample_class_assignments = np.random.randint(0, num_classes, size=num_samples)

    for i in range(num_samples):
        class_idx = sample_class_assignments[i]
        
        # Get the centroid for the assigned class
        text_centroid = text_class_centroids[class_idx]
        image_centroid = image_class_centroids[class_idx]
        
        # Add Gaussian noise to create variance within the class
        text_noise = np.random.normal(0, noise_scale, size=text_embedding_dim)
        image_noise = np.random.normal(0, noise_scale, size=image_embedding_dim)
        
        # Create the final embedding by adding noise to the centroid
        final_text_embedding = text_centroid + text_noise
        final_image_embedding = image_centroid + image_noise
        
        t_emb_list.append(final_text_embedding)
        i_emb_list.append(final_image_embedding)
        labels_list.append(class_idx)

    # Convert lists to numpy arrays
    t_emb = np.array(t_emb_list)
    i_emb = np.array(i_emb_list)
    labels = np.array(labels_list)
    # Normalize embeddings
    t_emb=normalize(t_emb)
    i_emb=normalize(i_emb)
    
    print("\nGeneration complete.")
    print(f"Shape of text embeddings (t_emb): {t_emb.shape}")
    print(f"Shape of image embeddings (i_emb): {i_emb.shape}")
    print(f"Shape of labels: {labels.shape}")

    return t_emb, i_emb, labels

def save_embeddings_to_npz(
    file_path: str,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    labels: np.ndarray
):
    """
    Saves the text embeddings, image embeddings, and labels to a single
    compressed.npz file.

    Args:
        file_path (str): The path to the output.npz file.
        text_embeddings (np.ndarray): The matrix of text embeddings.
        image_embeddings (np.ndarray): The matrix of image embeddings.
        labels (np.ndarray): The array of class labels.
    """
    print(f"\nSaving embeddings to {file_path}...")
    np.savez_compressed(
        file_path,
        report_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        labels=labels
    )
    print("Successfully saved embeddings.")

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters for the generation
    NUM_SAMPLES = 2000
    NUM_CLASSES = 2
    TEXT_EMBEDDING_DIM = 768  # Text embeddings will be in a 128-dim space
    IMAGE_EMBEDDING_DIM = 1024 # Image embeddings will be in a 256-dim space
    OUTPUT_NPZ_FILE = "Test_Syntetic.npz"

    # Generate the embedding matrices
    t_emb, i_emb, labels = generate_correlated_embeddings(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        latent_dim=32,       # A smaller shared space
        noise_scale=0.2      # Controls how "spread out" each class is
    )

    # Save the generated embeddings to a.npz file
    save_embeddings_to_npz(
        file_path=OUTPUT_NPZ_FILE,
        text_embeddings=t_emb,
        image_embeddings=i_emb,
        labels=labels
    )

    # --- Verification of the saved file ---
    # Load the data back from the.npz file to confirm it was saved correctly.
    print(f"\n--- Verifying the saved file: {OUTPUT_NPZ_FILE} ---")
    if os.path.exists(OUTPUT_NPZ_FILE):
        loaded_data = np.load(OUTPUT_NPZ_FILE)
        
        # Access arrays by the keys we provided during saving
        loaded_t_emb = loaded_data['report_embeddings']
        loaded_i_emb = loaded_data['image_embeddings']
        loaded_labels = loaded_data['labels']
        
        print("File loaded successfully. Verifying shapes:")
        print(f"  - Loaded text embeddings shape: {loaded_t_emb.shape}")
        print(f"  - Loaded image embeddings shape: {loaded_i_emb.shape}")
        print(f"  - Loaded labels shape: {loaded_labels.shape}")
        
        # Check if the loaded data matches the original data
        assert np.array_equal(t_emb, loaded_t_emb)
        assert np.array_equal(i_emb, loaded_i_emb)
        assert np.array_equal(labels, loaded_labels)
        print("Verification successful: Loaded data matches original data.")
    else:
        print(f"Error: File '{OUTPUT_NPZ_FILE}' was not found.")