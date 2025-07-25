import torch
import torch.nn as nn
from aeds import ConvAutoencoder  # Adjust this if your model is defined elsewhere

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    # Load the model architecture
    model = ConvAutoencoder()

    # Load saved weights
    state_dict = torch.load('aeds_model.pt', map_location='cpu')
    model.load_state_dict(state_dict)

    # Print model structure
    print("Model architecture:")
    print(model)

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Print layer-wise info
    print("\nLayer-wise parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name:40} {tuple(param.shape)}")

    # Optional: try a dummy forward pass
    try:
        dummy_input = torch.randn(1, 1, 128, 128)  # (B, C, H, W) assuming 1-channel input
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nForward pass successful. Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"\nForward pass failed: {e}")

if __name__ == '__main__':
    main()
