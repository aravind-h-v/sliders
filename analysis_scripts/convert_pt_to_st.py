import torch
import safetensors.torch
from pathlib import Path
import numpy as np
import json

# Global dictionary to store name mappings
layer_name_mappings = {}

def convert_layer_name(old_name):
    """Convert layer names from flux_slider format to flux_ostris format."""
    # If we've seen this name before, return the cached conversion
    if old_name in layer_name_mappings:
        return layer_name_mappings[old_name]
    
    new_name = old_name
    
    # Handle transformer blocks
    if old_name.startswith('lora_unet_transformer_blocks_'):
        
        parts = old_name.split('_')
        block_num = parts[4]  # Get block number
        # print(parts)
        # Convert lora weights
        if 'lora_up' in old_name:
            new_name = old_name.replace('lora_up', 'lora_B')
        elif 'lora_down' in old_name:
            new_name = old_name.replace('lora_down', 'lora_A')
        
        # print(new_name, block_num)
        # Replace prefix and convert to dot notation
        new_name = new_name.replace(
            f'lora_unet_transformer_blocks_{block_num}_attn_to_',
            f'transformer.transformer_blocks.{block_num}.attn.to_'
        ).replace(
            f'lora_unet_transformer_blocks_{block_num}_attn_add_',
            f'transformer.transformer_blocks.{block_num}.attn.add_'
        )
        # print(new_name)
        
    # Handle single transformer blocks
    elif old_name.startswith('lora_unet_single_transformer_blocks_'):
        parts = old_name.split('_')
        block_num = parts[5]  # Get block numbes
        # Convert lora weights
        if 'lora_up' in old_name:
            new_name = old_name.replace('lora_up', 'lora_B')
        elif 'lora_down' in old_name:
            new_name = old_name.replace('lora_down', 'lora_A')
        
        # print(new_name)
        # Replace prefix and convert to dot notation
        new_name = new_name.replace(
            f'lora_unet_single_transformer_blocks_{block_num}_attn_to_',
            f'transformer.single_transformer_blocks.{block_num}.attn.to_'
        ).replace(
            f'lora_unet_single_transformer_blocks_{block_num}_attn_add_',
            f'transformer.single_transformer_blocks.{block_num}.attn.add_'
        )
    
    new_name = new_name.replace('to_out_0', 'to_out')
    # Store mapping
    layer_name_mappings[old_name] = new_name
    return new_name

def save_name_mappings(output_path):
    """Save the layer name mappings to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(layer_name_mappings, f, indent=2)
    print(f"\nSaved layer name mappings to {output_path}")

def convert_pt_to_safetensors(pt_path, output_path=None, analyze=True):
    # Load the .pt file
    state_dict = torch.load(pt_path)
    
    # Create new state dict with converted names
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = convert_layer_name(key)
        new_state_dict[new_key] = value
        # raise Exception(f"new_key: {new_key}")
    
    # # Analyze the state dict if requested
    # if analyze:
    #     analyze_state_dict(new_state_dict, Path(pt_path).name)
    
    # If output path is not specified, use the same name but with .safetensors extension
    if output_path is None:
        output_path = str(Path(pt_path).with_suffix('.safetensors'))
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as safetensors
    safetensors.torch.save_file(new_state_dict, output_path)
    print(f"\nConverted {pt_path} to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to .pt or .safetensors file or directory")
    parser.add_argument("--output_path", type=str, help="Output path (optional)")
    
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if input_path.is_file():
        if input_path.suffix == '.pt':
            convert_pt_to_safetensors(str(input_path), args.output_path)
        else:
            print("Input file is already in safetensors format")

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_path = args.output_path.replace('.safetensors', '.json')
    save_name_mappings(mappings_path)
    
# python analysis_scripts/convert_pt_to_st.py --input_path flux-sliders/outputs/person-obese-mod/slider_0.pt --output_path outputs/person-obse-mode.safetensors
