import torch
from config import DeepSeekV3Config
from model import DeepSeekV3ForCausalLM
from mtp import DeepSeekV3WithMTP

def main():
    # 1. Initialize a small-scale configuration for demonstration
    config = DeepSeekV3Config.small_config()
    print(f"Initializing DeepSeek-V3 (Small Config)...")
    print(f"Model: {config.num_layers} layers, {config.hidden_size} hidden size")
    print(f"MoE: {config.num_experts} experts, {config.num_activated_experts} activated")
    
    # 2. Instantiate the model with MTP support
    model = DeepSeekV3WithMTP(config)
    model.eval()
    
    # 3. Dummy input for forward pass
    # [batch_size, sequence_length]
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    labels = torch.randint(0, config.vocab_size, (1, 32))
    
    print("\nRunning Forward Pass with MTP...")
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        
    logits = outputs["logits"]
    mtp_logits = outputs["mtp_logits"]
    loss = outputs["loss"]
    
    print(f"\nForward pass successful!")
    print(f"Main Logits Shape: {logits.shape}")
    print(f"MTP Logits Depth: {len(mtp_logits)}")
    for i, ml in enumerate(mtp_logits):
        print(f"  MTP Depth {i+1} Logits Shape: {ml.shape}")
    print(f"Calculated Total Loss (with MTP): {loss.item():.4f}")

    # 4. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters in this small config: {total_params / 1e6:.2f}M")

if __name__ == "__main__":
    main()
