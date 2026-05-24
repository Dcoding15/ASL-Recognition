from models import CNNLSTM, CNNBiLSTM, AttentionCNNBiLSTM, CNN3D, TCN, TwoStreamStatic, ViT, FrameFeatureEncoder
import torch
import torch.nn as nn
import os

dir = "C:\\Users\\HP\\PROJECTS\\ASL\\pre trained weigths\\"
model = [
    CNNLSTM(26), 
    CNNBiLSTM(26), 
    AttentionCNNBiLSTM(26), 
    CNN3D(26), 
    nn.Sequential(
        FrameFeatureEncoder(input_dim=3072, hidden_dim=512), 
        TCN(num_inputs=512, num_channels=[256, 256, 256], num_classes=26)
    ), 
    TwoStreamStatic(feature_dim=256, num_classes=26), 
    ViT(num_classes=26)
]
weights = [
    "cnn_lstm_weights.pth",
    "cnn_bi_lstm_weights.pth",
    "cnn_bi_lstm_attention_weights.pth", 
    "3d_cnn_weights.pth", 
    "tcn_weights.pth", 
    "two_stream_static_weights.pth", 
    "vit_weights.pth"
]

# Create output directory if it doesn't exist
output_dir = "model_parameters"
os.makedirs(output_dir, exist_ok=True)

for i in range(len(model)):
    # Load model weights
    model[i].load_state_dict(torch.load(f'{dir}{weights[i]}'))
    model[i].eval()  # Fixed: should be model[i] not model[1]
    
    # Create a clean filename from the weight name
    txt_filename = weights[i].replace('.pth', '_parameters.txt')
    txt_path = os.path.join(output_dir, txt_filename)
    
    # Create a summary filename
    summary_filename = weights[i].replace('.pth', '_summary.txt')
    summary_path = os.path.join(output_dir, summary_filename)
    
    print(f"Processing: {weights[i]}")
    print(f"Saving details to: {txt_path}")
    print(f"Saving summary to: {summary_path}")
    
    # Open files for writing
    with open(txt_path, 'w', encoding='utf-8') as f_details, \
         open(summary_path, 'w', encoding='utf-8') as f_summary:
        
        # Write header to both files
        header = f"Parameter Analysis for: {weights[i]}\n"
        header += f"Model Type: {type(model[i]).__name__}\n"
        header += "=" * 80 + "\n\n"
        
        f_details.write(header)
        f_summary.write(header)
        
        # Initialize counters for summary
        total_params = 0
        trainable_params = 0
        layer_counts = {}
        
        # Write parameter details
        f_details.write("DETAILED PARAMETER INFORMATION:\n")
        f_details.write("=" * 80 + "\n\n")
        
        for name, param in model[i].named_parameters():
            # Count parameters
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            
            # Track layer types
            layer_type = name.split('.')[0] if '.' in name else name
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + param_count
            
            # Write detailed information
            f_details.write(f"Parameter Name: {name}\n")
            f_details.write(f"Shape: {tuple(param.shape)}\n")
            f_details.write(f"Number of elements: {param_count:,}\n")
            f_details.write(f"Requires gradient: {param.requires_grad}\n")
            f_details.write(f"Data type: {param.dtype}\n")
            f_details.write(f"Mean value: {param.data.mean().item():.6f}\n")
            f_details.write(f"Std deviation: {param.data.std().item():.6f}\n")
            f_details.write(f"Min value: {param.data.min().item():.6f}\n")
            f_details.write(f"Max value: {param.data.max().item():.6f}\n")
            
            # Write first few values
            flat_values = param.data.flatten()[:10]
            f_details.write(f"First 10 values: {[f'{x:.6f}' for x in flat_values.tolist()]}\n")
            
            f_details.write("-" * 80 + "\n\n")
        
        # Write summary information
        f_summary.write("MODEL SUMMARY:\n")
        f_summary.write("=" * 80 + "\n\n")
        
        f_summary.write(f"Total parameters: {total_params:,}\n")
        f_summary.write(f"Trainable parameters: {trainable_params:,}\n")
        f_summary.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        f_summary.write(f"Percentage trainable: {(trainable_params/total_params*100):.2f}%\n\n")
        
        f_summary.write("PARAMETER DISTRIBUTION BY LAYER TYPE:\n")
        f_summary.write("-" * 60 + "\n")
        f_summary.write(f"{'Layer Type':<20} {'Parameters':<15} {'Percentage':<10}\n")
        f_summary.write("-" * 60 + "\n")
        
        for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_params * 100)
            f_summary.write(f"{layer_type:<20} {count:<15,} {percentage:>9.2f}%\n")
        
        f_summary.write("-" * 60 + "\n\n")
        
        # Write layer-by-layer summary
        f_summary.write("LAYER-BY-LAYER PARAMETER COUNT:\n")
        f_summary.write("=" * 80 + "\n")
        f_summary.write(f"{'Parameter Name':<50} {'Shape':<25} {'Params':<15}\n")
        f_summary.write("-" * 95 + "\n")
        
        for name, param in model[i].named_parameters():
            param_count = param.numel()
            f_summary.write(f"{name:<50} {str(tuple(param.shape)):<25} {param_count:<15,}\n")
        
        f_summary.write("-" * 95 + "\n")
        f_summary.write(f"{'TOTAL':<50} {'':<25} {total_params:<15,}\n")
    
    print(f"✓ Saved: {txt_filename}")
    print(f"✓ Saved: {summary_filename}")
    print()

print("=" * 80)
print("All parameter files have been saved to the 'model_parameters' directory!")