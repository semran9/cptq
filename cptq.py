import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
from tqdm.auto import tqdm
import gc
import json
import time
from contextlib import contextmanager
import torch.amp as amp  # Updated import
import copy
from qlinear import QuantizedLinear

@dataclass
class QuantConfig:
    model_name: str = "Qwen/Qwen1.5-0.5B"
    threshold: float = 0.05
    calib_batch_size: int = 4
    calib_steps: int = 50
    sequence_length: int = 512
    bits: int = 2
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    mixed_precision: bool = True

@contextmanager
def gpu_memory_check(name: str):
    """Context manager to track GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        yield
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        print(f"{name} memory change: {(mem_after - mem_before) / 1024**2:.2f}MB")
    else:
        yield

def evaluate(model, test_texts):
    """Evaluate model perplexity on test texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Load tokenizer if not already loaded
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Evaluating perplexity"):
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Calculate total loss and tokens
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def replace_linear_layers(model, steps=5000, batch=128):
    """
    Replace all Linear layers in the model with QuantizedLinear layers.
    
    Args:
        model: The PyTorch model to modify
        steps: Number of calibration steps for each QuantizedLinear layer
        batch: Batch size for calibration
    """
    # Keep track of replaced layers
    replaced_count = 0
    
    # Iterate through named modules
    for name, module in model.named_children():
        # If module is a Linear layer, replace it
        if isinstance(module, nn.Linear) and name not in "lm_head":
            setattr(model, name, QuantizedLinear(module, steps=steps, batch=batch))
            replaced_count += 1
            print(f"Replaced linear layer {name}")
        # If module has children, recursively replace their layers
        elif len(list(module.children())) > 0:
            replaced_count += replace_linear_layers(module, steps=steps, batch=batch)
    
    return replaced_count


def quantize_and_evaluate():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load calibration dataset from HuggingFace
    from datasets import load_dataset
    # Create tokenizer for calibration data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    
    # Create calibration dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length=512):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            text = self.texts[idx]
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze()
            }
    # Load test dataset from C4
    print("Loading test dataset from C4...")
    test_dataset = load_dataset("c4", "en", split="validation", streaming=True)
    test_texts = []
    for item in test_dataset:
        if len(item['text'].strip()) > 0:
            test_texts.append(item['text'])
            if len(test_texts) >= 50:
                break
    print(f"Loaded {len(test_texts)} text samples for testing")

    # Create test dataset
    test_dataset = TextDataset(test_texts, tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,  # Same batch size as calibration for consistency
        shuffle=False,  # No need to shuffle test data
        num_workers=2
    )
    
    print(f"Created test dataloader with {len(test_loader)} batches")

    # Load pretrained model
    print("\nLoading pretrained Qwen model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B", 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    original_model.eval()
    print("Model loaded successfully")

    quantized_model = copy.deepcopy(original_model)
    count = replace_linear_layers(quantized_model)

    # Evaluate both models
    print("\nEvaluating models...")
    orig_acc = evaluate(original_model, test_loader, device)
    quant_acc = evaluate(quantized_model, test_loader, device)

    # Calculate model sizes
    orig_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
    quant_size = sum(p.numel() * 2 / 8 / (1024 * 1024) for p in quantized_model.parameters())

    # Print results
    print("\nResults:")
    print(f"Original Model:")
    print(f"  Accuracy: {orig_acc:.2f}%")
    print(f"  Model size: {orig_size:.2f}MB")

    print(f"\nQuantized Model:")
    print(f"  Accuracy: {quant_acc:.2f}%")
    print(f"  Model size: {quant_size:.2f}MB")
    print(f"  Size reduction: {(1 - quant_size/orig_size)*100:.1f}%")

    # Analyze quantization statistics
    print("\nQuantization Statistics:")
    for idx, layer in enumerate(quantized_model.quantized_layers):
        with torch.no_grad():
            quantized_weight, _ = layer.quantizer(layer.weight)
            sparsity = torch.mean((torch.abs(quantized_weight) < 1e-6).float()).item()
            pos_weights = torch.mean((quantized_weight > 1e-6).float()).item()
            neg_weights = torch.mean((quantized_weight < -1e-6).float()).item()

            print(f"\nLayer {idx+1}:")
            print(f"Sparsity: {sparsity:.2%}")
            print(f"Positive weights: {pos_weights:.2%}")
            print(f"Negative weights: {neg_weights:.2%}")

    return original_model, quantized_model

if __name__ == "__main__":
    original_model, quantized_model = quantize_and_evaluate()
