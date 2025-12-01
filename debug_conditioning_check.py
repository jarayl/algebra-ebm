
import torch
import numpy as np
from algebra_encoder import create_character_encoder
from algebra_models import AlgebraEBM

def check_encoder_diversity():
    print("Checking Encoder Diversity...")
    encoder = create_character_encoder(d_model=128)
    
    eq1 = "4*x-3*x=114"
    eq2 = "10*x-9*x=-18"
    eq3 = "2*x-1*x=96"
    
    emb1 = encoder.encode_equation_string(eq1)
    emb2 = encoder.encode_equation_string(eq2)
    emb3 = encoder.encode_equation_string(eq3)
    
    dist12 = torch.norm(emb1 - emb2).item()
    dist13 = torch.norm(emb1 - emb3).item()
    dist23 = torch.norm(emb2 - emb3).item()
    
    print(f"Distance between '{eq1}' and '{eq2}': {dist12}")
    print(f"Distance between '{eq1}' and '{eq3}': {dist13}")
    print(f"Distance between '{eq2}' and '{eq3}': {dist23}")
    
    if dist12 < 1e-5 or dist13 < 1e-5 or dist23 < 1e-5:
        print("WARNING: Encoder collapse detected! Embeddings are identical.")
    else:
        print("Encoder looks okay (embeddings are distinct).")

def check_model_conditioning():
    print("\nChecking Model Conditioning Sensitivity...")
    # Initialize model
    ebm = AlgebraEBM(inp_dim=128, out_dim=128)
    ebm.eval()
    
    # Create dummy inputs
    batch_size = 1
    inp1 = torch.randn(batch_size, 128)
    inp2 = torch.randn(batch_size, 128) # Different input
    out = torch.randn(batch_size, 128)  # Fixed output
    t = torch.zeros(batch_size).long()
    
    # Compute energy
    with torch.no_grad():
        energy1 = ebm(inp1, out, t)
        energy2 = ebm(inp2, out, t)
        
    print(f"Energy with inp1: {energy1.item()}")
    print(f"Energy with inp2: {energy2.item()}")
    
    diff = torch.abs(energy1 - energy2).item()
    print(f"Energy difference: {diff}")
    
    if diff < 1e-5:
        print("WARNING: Model ignores conditioning input! Energy is identical.")
    else:
        print("Model is sensitive to conditioning input.")

if __name__ == "__main__":
    check_encoder_diversity()
    check_model_conditioning()
