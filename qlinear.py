import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
def weight_quant(w):
    """Per-tensor quantization to 1.58 bits"""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u

# def weight_quant(w):
#     scale = 1.0 / (2 * w.std().clamp(min=1e-5))
#     u = (w * scale).round().clamp_(-1, 1)
#     return u


class QuantizedLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bits: int = 4, 
        bias: bool = True,
        norm_bits: int = 8  # Separate bits for norm quantization
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.norm_bits = norm_bits
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Quantization state
        self.quantized = False
        self.H = None
        self.quantized_weight = None
        self.quantized_norms = None
        self.quantized_directions = None
        
    def compute_H(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hessian approximation using calibration data."""
        with torch.no_grad():
            H = torch.einsum('bi,bj->ij', x, x) / x.shape[0]
            H.diagonal().add_(1e-8)
            return H
    
    def compute_angular_similarity(self, row1: torch.Tensor, row2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two vectors."""
        norm1 = torch.norm(row1)
        norm2 = torch.norm(row2)
        if norm1 == 0 or norm2 == 0:
            return torch.tensor(0.0, device=row1.device)
        return torch.dot(row1, row2) / (norm1 * norm2)
    
    def quantize_weight(self, x_cal: torch.Tensor) -> None:
        """Quantize weights using direction-aware GPTQ algorithm."""
        if self.quantized:
            return
            
        # Compute Hessian approximation
        self.H = self.compute_H(x_cal.reshape(x_cal.shape[0]*x_cal.shape[1], x_cal.shape[-1]))
        
        # Get original weights
        W = self.weight.data
        
        # Compute and quantize row norms separately
        row_norms = torch.norm(W, p=2, dim=1)
        self.quantized_norms = row_norms
        
        # Normalize rows to get directions
        W_normalized = W / row_norms.unsqueeze(1).clamp(min=1e-8)
        
        # Initialize quantized directions and error
        W_q_directions = W_normalized # torch.zeros_like(W_normalized)
        E = torch.zeros_like(W_normalized)

        
        # Compute importance scores
        importance = torch.norm(W_normalized @ self.H, p=2, dim=1)
        quant_imp = weight_quant(W_normalized)
        importance = torch.diag(W_normalized @ quant_imp.T, 0)
        ordered_indices = torch.argsort(importance, descending=True)
        
        # Quantize directions in order of importance
        for idx in ordered_indices:
            # Current direction with accumulated error
            w_dir = W_normalized[idx] + E[idx]
            w_dir = w_dir / torch.norm(w_dir).clamp(min=1e-8)  # Re-normalize
            
            # Quantize the direction
            w_q_dir = self.quantize_direction(w_dir)
            W_q_directions[idx] = w_q_dir
            
            # Compute and propagate error
            q_error = (W_normalized[idx] - w_q_dir)
            error = q_error @ self.H
            
            # Distribute error to remaining directions
            remaining_mask = torch.zeros_like(importance, dtype=torch.bool)
            remaining_mask[ordered_indices[ordered_indices > idx]] = True
            if remaining_mask.any():
                E[remaining_mask] += error.unsqueeze(0) / remaining_mask.sum()
        
        # Store quantized directions
        self.quantized_directions = W_q_directions
        
        # Combine quantized norms and directions
        self.quantized_weight = self.quantized_directions * self.quantized_norms.unsqueeze(1)
        self.quantized_weight = nn.Parameter(self.quantized_weight.to(self.weight.device))
        print(self.quantized_weight.device)
        self.bias =  nn.Parameter(self.bias.to(self.weight.device))
        self.quantized = True
        del self.H
        del self.weight
    
    def quantize_direction(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize a normalized direction vector."""
        # Scale to use full range of quantization bits
        # Compute scale for uniform quantization
        # max_val = w.abs().max().clamp(min=1e-8)  # Avoid division by zero
        # scale = max_val / (2 ** (self.bits - 1) - 1)

        # Normalize and quantize
        # w_scaled = w / scale
        # w_q = torch.round(w_scaled).clamp(-2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)

        # Rescale quantized values back to the original range
        # w_q = w_q * scale

        # Renormalize to ensure unit length
        ### EDITS ####
        w_q = weight_quant(w)

        norm = torch.norm(w_q).clamp(min=1e-8)
        return w_q / norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using quantized or full-precision weights."""
        if not self.quantized:
            self.quantize_weight(x)
        weight = self.quantized_weight
        return F.linear(x, weight, self.bias)

def test_directional_quantization():
    """Test the direction-aware quantization."""
    torch.manual_seed(0)
    
    def compute_metrics(W_orig: torch.Tensor, W_quant: torch.Tensor):
        """Compute comprehensive metrics."""
        # Normalize both matrices row-wise for direction comparison
        W_orig_norm = W_orig / torch.norm(W_orig, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        W_quant_norm = W_quant / torch.norm(W_quant, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        
        # Compute metrics
        direction_mse = F.mse_loss(W_orig_norm, W_quant_norm)
        weight_mse = F.mse_loss(W_orig, W_quant)
        
        # Compute norm preservation
        orig_norms = torch.norm(W_orig, p=2, dim=1)
        quant_norms = torch.norm(W_quant, p=2, dim=1)
        norm_mse = F.mse_loss(orig_norms, quant_norms)
        
        return {
            'direction_mse': direction_mse.item(),
            'weight_mse': weight_mse.item(),
            'norm_mse': norm_mse.item()
        }
    
    # Test configuration
    in_features, out_features = 768, 3072
    
    for bits in [2, 3, 4]:
        print(f"\nTesting {bits}-bit quantization:")
        
        # Create layer
        layer = QuantizedLinear(in_features, out_features, bits=bits)
        W_orig = layer.weight.data.clone()
        
        # Generate calibration data
        x_cal = torch.randn(128, in_features)
        
        # Quantize
        layer.quantize_weight(x_cal)
        
        # Compute all metrics
        metrics = compute_metrics(W_orig, layer.quantized_weight)
        
        # Test forward pass
        x = torch.randn(32, in_features)
        with torch.no_grad():
            out_orig = F.linear(x, W_orig)
            out_quant = layer(x)
            output_mse = F.mse_loss(out_orig, out_quant)
        
        print(f"Direction MSE: {metrics['direction_mse']:.6f}")
        print(f"Weight MSE: {metrics['weight_mse']:.6f}")
        print(f"Norm MSE: {metrics['norm_mse']:.6f}")
        print(f"Output MSE: {output_mse:.6f}")
        
        # Count unique values
        unique_values = torch.unique(layer.quantized_weight)
        print(f"Number of unique values: {len(unique_values)}")

#if __name__ == "__main__":
#    test_directional_quantization()
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt

def compute_metrics(original: torch.Tensor, quantized: torch.Tensor) -> Dict[str, float]:
    """
    Compute various error metrics between original and quantized tensors.
    """
    mse = F.mse_loss(original, quantized).item()
    mae = torch.mean(torch.abs(original - quantized)).item()
    max_error = torch.max(torch.abs(original - quantized)).item()
    
    # Compute relative error (avoiding division by zero)
    rel_error = torch.abs(original - quantized) / (torch.abs(original) + 1e-8)
    mean_rel_error = torch.mean(rel_error).item()
    
    # Compute PSNR
    max_val = max(original.abs().max().item(), quantized.abs().max().item())
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'mean_rel_error': mean_rel_error,
        'psnr': psnr
    }

def test_layer_behavior(
    in_features: int,
    out_features: int,
    bits: int,
    batch_size: int = 32,
    cal_size: int = 128
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Test quantization effects on both weights and layer outputs.
    """
    # Create layer and save original weights
    layer = QuantizedLinear(in_features, out_features, bits=bits)
    W_orig = layer.weight.data.clone()
    
    # Generate calibration data
    x_cal = torch.randn(cal_size, in_features)
    
    # Quantize
    layer.quantize_weight(x_cal)
    
    # Compute weight metrics
    weight_metrics = compute_metrics(W_orig, layer.quantized_weight)
    
    # Test forward pass behavior
    x_test = torch.randn(batch_size, in_features)
    with torch.no_grad():
        out_orig = F.linear(x_test, W_orig)
        out_quant = layer(x_test)
        output_metrics = compute_metrics(out_orig, out_quant)
    
    return weight_metrics, output_metrics

def visualize_quantization_effects(original: torch.Tensor, quantized: torch.Tensor):
    """
    Create visualization of quantization effects.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot weight distributions
    plt.subplot(131)
    plt.hist(original.flatten().numpy(), bins=50, alpha=0.5, label='Original')
    plt.hist(quantized.flatten().numpy(), bins=50, alpha=0.5, label='Quantized')
    plt.title('Weight Distributions')
    plt.legend()
    
    # Plot quantization error distribution
    plt.subplot(132)
    error = (original - quantized).flatten().numpy()
    plt.hist(error, bins=50)
    plt.title('Quantization Error Distribution')
    
    # Plot original vs quantized scatter
    plt.subplot(133)
    plt.scatter(original.flatten().numpy(), quantized.flatten().numpy(), alpha=0.1)
    plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect correlation line
    plt.title('Original vs Quantized Values')
    plt.xlabel('Original')
    plt.ylabel('Quantized')
    
    plt.tight_layout()
    return plt.gcf()

def comprehensive_test_suite():
    """
    Run comprehensive tests across different configurations.
    """
    # Test configurations
    configs = [
        # (in_features, out_features, bits)
        (768, 3072, 2),  # Common transformer MLP expansion, aggressive quantization
        (768, 3072, 3),  # Same size, more bits
        (768, 3072, 4),  # Same size, even more bits
        (512, 512, 2),   # Square matrix
        (1024, 256, 2),  # Dimension reduction
    ]
    
    results = {}
    
    print("\nRunning comprehensive quantization tests...")
    print("-" * 80)
    
    for in_features, out_features, bits in configs:
        print(f"\nTesting configuration: {in_features} -> {out_features}, {bits} bits")
        
        # Run test multiple times to account for randomness
        weight_metrics_list = []
        output_metrics_list = []
        
        for seed in range(3):  # Run 3 times with different seeds
            torch.manual_seed(seed)
            weight_metrics, output_metrics = test_layer_behavior(in_features, out_features, bits)
            weight_metrics_list.append(weight_metrics)
            output_metrics_list.append(output_metrics)
        
        # Average metrics across runs
        avg_weight_metrics = {k: np.mean([m[k] for m in weight_metrics_list]) for k in weight_metrics_list[0]}
        avg_output_metrics = {k: np.mean([m[k] for m in output_metrics_list]) for k in output_metrics_list[0]}
        
        # Store results
        config_key = f"in{in_features}_out{out_features}_bits{bits}"
        results[config_key] = {
            'weight_metrics': avg_weight_metrics,
            'output_metrics': avg_output_metrics
        }
        
        # Print results
        print("\nWeight Metrics:")
        for k, v in avg_weight_metrics.items():
            print(f"{k:15s}: {v:.6f}")
        
        print("\nOutput Metrics:")
        for k, v in avg_output_metrics.items():
            print(f"{k:15s}: {v:.6f}")
        
        # Create and store visualization for one instance
        # torch.manual_seed(42)
        layer = QuantizedLinear(in_features, out_features, bits=bits)
        W_orig = layer.weight.data.clone()
        x_cal = torch.randn(128, in_features)
        layer.quantize_weight(x_cal)
        
        fig = visualize_quantization_effects(W_orig, layer.quantized_weight)
        plt.title(f"Quantization Effects ({in_features}->{out_features}, {bits} bits)")
        plt.close(fig)
    
    return results

def analyze_bit_width_impact():
    """
    Analyze how different bit widths affect quantization quality.
    """
    print("\nAnalyzing bit width impact...")
    print("-" * 80)
    
    in_features, out_features = 768, 3072
    bit_widths = [2, 3, 4, 5]
    
    mse_values = []
    psnr_values = []
    
    for bits in bit_widths:
        weight_metrics, output_metrics = test_layer_behavior(in_features, out_features, bits)
        mse_values.append(weight_metrics['mse'])
        psnr_values.append(weight_metrics['psnr'])
        
        print(f"\nBit width: {bits}")
        print(f"MSE: {weight_metrics['mse']:.6f}")
        print(f"PSNR: {weight_metrics['psnr']:.2f} dB")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(bit_widths, mse_values, 'o-')
    plt.title('MSE vs Bit Width')
    plt.xlabel('Bits')
    plt.ylabel('MSE')
    plt.yscale('log')
    
    plt.subplot(122)
    plt.plot(bit_widths, psnr_values, 'o-')
    plt.title('PSNR vs Bit Width')
    plt.xlabel('Bits')
    plt.ylabel('PSNR (dB)')
    
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    # Run all tests
    all_results = comprehensive_test_suite()
    
    # Analyze bit width impact
    bit_width_analysis = analyze_bit_width_impact()
    
    print("\nTesting completed!")
    print("-" * 80)

