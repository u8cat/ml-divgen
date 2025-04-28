import os
import random
import pickle
from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the data
post_samples_path = 'model_outputs/ContextualAI_archangel_sft_pythia2-8b_euclaise_writingprompts_validation_samples_100.pkl'
base_samples_path = 'model_outputs/EleutherAI_pythia-2.8b_euclaise_writingprompts_validation_samples_100.pkl'

with open(post_samples_path, 'rb') as f:
    post_samples = pickle.load(f)
with open(base_samples_path, 'rb') as f:
    base_samples = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')

def plot_top_k_distribution_comparison(sample_a, sample_b, tokenizer, token_pos, k=10, sample_a_name='Sample A', sample_b_name='Sample B'):
    """
    Compare the top k token prob distribution of two samples.
    Note: The distribution plot might have [k, 2k] tokens, 
          these are combined top-k logits from both samples.
    """
    probs_a = sample_a['generated_logits'].softmax(dim=-1)[token_pos].unsqueeze(0)
    probs_b = sample_b['generated_logits'].softmax(dim=-1)[token_pos].unsqueeze(0)

    top_k_a = probs_a.topk(k, dim=-1).indices
    top_k_b = probs_b.topk(k, dim=-1).indices

    # combine to get all the token ids (unique)
    combined_top_k = torch.unique(torch.cat([top_k_a, top_k_b], dim=-1)).tolist()
    combined_probs_a = probs_a[0, combined_top_k].numpy()
    combined_probs_b = probs_b[0, combined_top_k].numpy()
    
    # Get token strings for visualization
    token_labels = [repr(tokenizer.decode(token_id)) for token_id in combined_top_k]
    
    # Sort by average probability to make the plot more readable
    avg_probs = (combined_probs_a + combined_probs_b) / 2
    sort_indices = np.argsort(avg_probs)[::-1]  # descending order
    
    combined_probs_a = combined_probs_a[sort_indices]
    combined_probs_b = combined_probs_b[sort_indices]
    token_labels = [token_labels[i] for i in sort_indices]
    combined_top_k = [combined_top_k[i] for i in sort_indices]
    
    # Plotting
    plt.figure(figsize=(16, 6))
    x = np.arange(len(combined_top_k))
    width = 0.35
    
    plt.bar(x - width/2, combined_probs_a, width, label=sample_a_name)
    plt.bar(x + width/2, combined_probs_b, width, label=sample_b_name)
    plt.grid(True, axis='y')
    plt.ylabel('Probability')
    plt.xlabel('Tokens')
    plt.title(f'Top-{k} token probability distribution comparison')
    plt.xticks(x, token_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

# Create imgs directory if it doesn't exist
os.makedirs('imgs', exist_ok=True)

# Run 50 random samples
for i in range(50):
    # Randomly select sample index and token position
    sample_idx = random.randint(0, len(post_samples)-1)
    token_pos = random.randint(0, 51)  # Assuming max token length is 100
    
    # Create figure
    plt.figure(figsize=(16, 6))
    
    # Plot comparison
    plot_top_k_distribution_comparison(
        post_samples[sample_idx], 
        base_samples[sample_idx], 
        tokenizer,
        token_pos=token_pos,
        k=10,
        sample_a_name='SFT Model',
        sample_b_name='Base Model'
    )
    
    # Save figure with specified naming format
    plt.savefig(f'imgs/idx_{sample_idx}_pos_{token_pos}.png')
    plt.close()
    
    print(f'Saved image {i+1}/50: idx_{sample_idx}_pos_{token_pos}.png') 