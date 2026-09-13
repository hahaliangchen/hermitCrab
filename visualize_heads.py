import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: This script requires 'matplotlib' and 'seaborn'.")
    print("Please install them via: pip install matplotlib seaborn")
    sys.exit(1)

# Configure Matplotlib for Chinese Support (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display

from bert_simple.model import BertForMaskedLM, BertConfig
from bert_simple.tokenizer import SimpleBertTokenizer

def load_trained_model(model_dir):
    print(f"Loading model from {model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer using the simple logic (vocab.json)
    tokenizer = SimpleBertTokenizer.from_pretrained(model_dir)
    
    # Load Model
    # Since we modified the code, we need to make sure we load weights correctly.
    # The saved weights are standard keys, so adding output_attentions logic shouldn't break loading.
    model = BertForMaskedLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def visualize_attention(text, model_path="outputs/simple-tokenizer-mlm"):
    # If using default relative path, make sure it exists
    if not os.path.exists(model_path):
        # Fallback to absolute path just in case user runs from weird CWD, or check common spots
        abs_path = os.path.join(os.path.dirname(__file__), "outputs", "simple-tokenizer-mlm")
        if os.path.exists(abs_path):
            model_path = abs_path
        else:
            # Fallback to the known absolute path from previous sessions just to be safe
            model_path = "d:\\project\\bert-simple\\outputs\\simple-tokenizer-mlm"

    model, tokenizer, device = load_trained_model(model_path)
    
    # Prepare Input
    print(f"\nAnalyzing Sentence: {text}")
    enc = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    
    # Forward Pass with Attention Output
    with torch.no_grad():
        # returns: (logits, all_attentions) 
        # all_attentions is a tuple of (batch, num_heads, seq_len, seq_len) for each layer
        outputs = model(input_ids, output_attentions=True)
        all_attentions = outputs[-1] # The last returned element is all_attentions

    # Get the last layer's attention
    # Shape: [1, Num_Heads, Seq_Len, Seq_Len]
    last_layer_attn = all_attentions[-1].cpu()
    num_heads = last_layer_attn.size(1)
    
    print(f"Visualizing Last Layer (Layer {len(all_attentions)}) - {num_heads} Heads")
    
    # Plotting
    # Create a grid of subplots
    cols = 2
    rows = (num_heads + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = axes.flatten()
    
    for i in range(num_heads):
        ax = axes[i]
        # Get attention matrix for Head i
        # Shape: [Seq_Len, Seq_Len]
        attn_data = last_layer_attn[0, i]
        
        sns.heatmap(
            attn_data, 
            ax=ax, 
            cmap="viridis", 
            xticklabels=tokens, 
            yticklabels=tokens,
            annot=False,  # Turn on if you want numbers, but messy for long text
            square=True,
            cbar=True
        )
        ax.set_title(f"Head {i+1} (Function ???)")
        ax.set_xlabel("Key (Target)")
        ax.set_ylabel("Query (Source)")
        
        # Rotate x labels for Chinese
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.suptitle(f"Attention Map: '{text[:15]}...'", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    save_path = "attention_heatmap.png"
    plt.savefig(save_path, dpi=120)
    print(f"✅ Visualization saved to {os.path.abspath(save_path)}")
    # Also show if environment supports it (not usually in this headless agent mode)
    # plt.show()

if __name__ == "__main__":
    # Test Sentence (Related to the training corpus)
    sentence = "项羽派兵攻打汉王"
    visualize_attention(sentence)
