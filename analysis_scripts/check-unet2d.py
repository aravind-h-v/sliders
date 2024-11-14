import torch
from diffusers import FluxTransformer2DModel
# Create an instance of the model first

pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
weight_dtype = torch.bfloat16
transformer = FluxTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer", 
    torch_dtype=weight_dtype
)

# Now iterate through the named modules of the instance
for name, module in transformer.named_modules():
    # if 'to_out' in name:
    print(name)
