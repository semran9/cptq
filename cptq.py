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

class BiOrthogonalRotation(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        # Initialize with small random values
        self.U_skew = nn.Parameter(torch.randn(rows, rows) * 0.01)
        self.V_skew = nn.Parameter(torch.randn(cols, cols) * 0.01)

    def get_orthogonal(self, A_skew):
        A_skew = (A_skew - A_skew.t()) / 2
        I = torch.eye(A_skew.shape[0], device=A_skew.device)
        return torch.matmul(I + A_skew, torch.inverse(I - A_skew))

    def forward(self):
        U = self.get_orthogonal(self.U_skew)
        V = self.get_orthogonal(self.V_skew)
        return U, V

class LayerTernaryQuantizer(nn.Module):
    def __init__(self, weight_shape, threshold=0.05):
        super().__init__()
        self.rotation = BiOrthogonalRotation(weight_shape[0], weight_shape[1])
        self.threshold = threshold
        self.register_buffer('scale', torch.ones(1))

    def normalize_matrix(self, M):
        row_norms = torch.sqrt(torch.sum(M ** 2, dim=1, keepdim=True))
        row_norms = torch.where(row_norms > 0, row_norms, torch.ones_like(row_norms))
        self.scale = row_norms
        return M / (row_norms + 1e-8)

    def quantize(self, M):
        return torch.where(torch.abs(M) > self.threshold,
                         torch.sign(M),
                         torch.zeros_like(M))

    def forward(self, M):
        M_norm = self.normalize_matrix(M)
        U, V = self.rotation()
        V = V.to(M.dtype)
        M_norm = M_norm.to(M.dtype)
        M_rotated = torch.matmul(torch.matmul(U.to(M.dtype), M_norm), V.t())
        M_quantized = self.quantize(M_rotated)
        M_quantized = M_quantized * self.scale
        # print(M_quantized)
        return M_quantized, (U, V)

class QuantizedLinear(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Keep original weights frozen
        self.register_buffer('weight', original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data.clone())
        else:
            self.register_buffer('bias', None)

        # Only the rotation parameters are trainable
        self.quantizer = LayerTernaryQuantizer(self.weight.shape).to(original_layer.weight.device)

        # Initialize quantized weights
        with torch.no_grad():
            self.quantized_weight, _ = self.quantizer(self.weight)

    def forward(self, x):
        if self.training:
            # During calibration
            quantized_weight, _ = self.quantizer(self.weight)
        else:
            # During inference
            quantized_weight = self.quantized_weight
        return F.linear(x, quantized_weight, self.bias)

    def update_quantized_weight(self):
        with torch.no_grad():
            self.quantized_weight, _ = self.quantizer(self.weight)

class QuantizedBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        for name, module in original_block.named_children():
            if isinstance(module, nn.Linear):
                quantized_layer = QuantizedLinear(module)
                quantized_layer.is_training = True
                setattr(self, name, quantized_layer)
            else:
                setattr(self, name, module)
        self.block = original_block

def quantize_block(original_block):
    for name, module in original_block.named_children():
        if isinstance(module, nn.Linear):
            # Replace with quantized linear layer
            quantized_layer = QuantizedLinear(module).to(module.weight.device)
            quantized_layer.is_training = True
            setattr(original_block, name, quantized_layer)
        else:
            # Recursively quantize submodules, if any
            quantize_block(module)
    # return original_block

def calibrate_rotations(teacher, student, idx, calib_loader, device, epochs=5):
    """Calibrate rotation matrices using a small calibration set"""
    block = student.model.layers[idx]
    student.zero_grad()
    optimizer = torch.optim.Adam(block.parameters(), lr=0.001)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        student.train()  # Enable training mode for rotation updates
        total_loss = 0

        for batch in tqdm(calib_loader, desc=f'Calibration Epoch {epoch+1}'):
            batch = {k: v.to(teacher.device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Get teacher output from next block
            with torch.no_grad():
                teacher_output = teacher(**batch, output_hidden_states = True)

            # Get current block output 
            student_output = student(**batch, output_hidden_states = True)

            for count, (sh, th) in enumerate(zip(student_output.hidden_states, teacher_output.hidden_states)):
                # print(count)
                if count >= idx+1:
                    # Calculate MSE loss between outputs
                    loss = F.mse_loss(sh, th)
                    # print(loss)
                    pass
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            

        avg_loss = total_loss / len(calib_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {name: param.clone() for name, param in block.state_dict().items()}

    # Restore best state
    block.load_state_dict(best_state)

    # Update quantized weights for inference
    for name, layer in block.named_modules():
        if isinstance(layer, QuantizedLinear):
            layer.update_quantized_weight()
    block.eval()  # Set to eval mode for inference

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

def quantize_and_evaluate():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load calibration dataset from HuggingFace
    from datasets import load_dataset
    
    print("Loading calibration dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Convert text to list of samples
    calib_texts = []
    for item in dataset:
        # Split into chunks of reasonable length
        text = item['text']
        if len(text.strip()) > 0:  # Skip empty lines
            calib_texts.append(text)
            if len(calib_texts) >= 1000:  # Limit number of samples
                break
                
    print(f"Loaded {len(calib_texts)} text samples for calibration")
    # Create tokenizer for calibration data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    
    # Create calibration dataset
    class CalibrationDataset(torch.utils.data.Dataset):
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
    
    # Create dataloader
    calib_dataset = CalibrationDataset(calib_texts, tokenizer)
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=8,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=2
    )
    
    print(f"Created calibration dataloader with {len(calib_loader)} batches")

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
    test_dataset = CalibrationDataset(test_texts, tokenizer)
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
    for idx, layer in enumerate(quantized_model.model.layers):
        # Quantize layer using quantized block
        quantize_block(
            layer
        )
        # print(quantized_model)
        # Calibrate rotations
        print("\nCalibrating rotation matrices...")
        calibrate_rotations(original_model, quantized_model, idx, calib_loader, device)

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
