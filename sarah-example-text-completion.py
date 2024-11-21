import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

gpu_id = 1
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if gpu_id >= torch.cuda.device_count():
    raise RuntimeError(f"GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}")

torch.cuda.set_device(gpu_id)
device = f'cuda:{gpu_id}'
print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")

# Load tokenizer and model from Hugging Face cache (~/.cache)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading model and tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prompt customization
input_prompt = input("Enter your custom input prompt: ")

# Tokenize input
encoded_inputs = tokenizer(input_prompt, return_tensors="pt", padding=True).to(device)
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

# Print input length
prompt_context_length = len(input_ids[0])
print(f"Input prompt context length: {prompt_context_length}")

# Generate text (without return_dict or attention outputs)
print("Generating text...")
max_new_tokens = 100
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2
)

# Decode the generated text
print("generated ", len(generated_ids), " number of tokens")
decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Text:")
print(decoded_text)

# Extract attention scores using the forward pass
print("\nExtracting attention scores...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True  # Enable attention outputs
    )

# Access attention scores
attentions = outputs.attentions  # List of tensors, one per layer
print(f"Number of layers with attention: {len(attentions)}")

# Print attention score shapes and percentage of zeros
for i, layer_attention in enumerate(attentions):
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    print(f"Layer {i + 1} attention shape: {layer_attention.shape}")
    
    # Calculate the total number of values and zeros
    total_values = layer_attention.numel()  # Total number of elements
    zero_values = (layer_attention == 0).sum().item()  # Count of zero values
    
    # Calculate percentage of zeros
    zero_percentage = (zero_values / total_values) * 100

    row_sums = layer_attention.sum(dim=-1)  # Shape: (batch_size, num_heads, seq_len)
    row_sum_deviation = (row_sums - 1).abs().mean().item()  # Mean deviation from 1

    print(f"Layer {i + 1}: {zero_values}/{total_values} values are zero ({zero_percentage:.2f}%)")
    print(f"Layer {i + 1}: Average row sum deviation from 1: {row_sum_deviation:.6f}")

    # print(layer_attention)
    print(row_sums[0, 0, :50])
    print(f"Layer {i + 1}: {zero_values}/{total_values} values are zero ({zero_percentage:.2f}%)")
