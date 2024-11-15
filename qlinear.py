import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def normalize_vector(vector):
    return vector / torch.linalg.norm(vector, dim=-1, keepdim=True)

def weight_quant(w):
    """Per-tensor quantization to 1.58 bits"""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u

class MatrixMultApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.2))
        self.bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, V, V_p, X):
        """
        Args:
            V: Original weight matrix [out_features, in_features]
            V_p: Quantized weight matrix [out_features, in_features]
            X: Input matrix [batch_size, in_features]
        """
        # Compute tangent matrix
        T = (V - V_p * torch.cos(self.theta)) / torch.sin(self.theta)
        
        # Compute matrix products for the entire batch at once
        X_T = torch.matmul(X, weight_quant(T).T)  # [batch_size, out_features]
        V_p_X = torch.matmul(X, V_p.T)  # [batch_size, out_features]
        
        # Get signs for all outputs at once
        sign = torch.sign(X_T * 100).detach()  # [batch_size, out_features]
        
        # Compute approximation for all outputs
        appx = torch.cos(self.theta) * V_p_X + sign * self.bias
        return appx
    def return_parameters(self, V, V_p):
        T = (V - V_p * torch.cos(self.theta)) / torch.sin(self.theta)
        return weight_quant(T), self.theta, self.bias

class QuantizedLinear(nn.Linear):
    def __init__(self, input_layer, steps = 5000, batch = 128):
        V = input_layer.weight.detach()
        V.requires_grad = False
        out_features, in_features = V.shape
        self.out_features = out_features
        super().__init__(in_features, out_features, True)
        if out_features == in_features:
            V_p = torch.eye(in_features)
        else:
            if out_features//in_features == 0:
                V_p = torch.eye(in_features)[:out_features, :]
            else:
                numcats = out_features//in_features + 1
                V_p = []
                for i in range(numcats):
                    V_p.append(torch.eye(in_features))
                V_p = torch.cat(V_p, dim = 0)
                V_p = V_p[:out_features, :]
        print(V.shape, V_p.shape)
        self.v_norm = torch.norm(V, dim=-1, keepdim=True).T
        self.approximator = MatrixMultApproximator()
        self.calibrate(V, V_p, steps, batch)
        self.T, self.theta, self.sub_bias = self.approximator.return_parameters(V, V_p)
         # Register parameters
        del self.approximator
        del self.weight
        self.T = nn.Parameter(self.T)
        self.theta = nn.Parameter(self.theta)
        self.sub_bias = nn.Parameter(self.sub_bias)
    
    def calibrate(self, V, V_p, n_epochs, batch_size):
        """Calibrate the matrix multiplication approximator"""
        optimizer = optim.AdamW(self.approximator.parameters(), lr=0.1)
        pbar = tqdm(range(n_epochs), desc=("Calibrating..."))
        for epoch in pbar:
            # Generate random input vectors
            X = torch.randn(batch_size, self.in_features)
            
            # Store original norms
            v_norm = torch.norm(V, dim=1, keepdim=True)  # [out_features, 1]
            x_norm = torch.norm(X, dim=1, keepdim=True)  # [batch_size, 1]
            
            # Normalize vectors without changing original V and V_p
            X_normalized = normalize_vector(X)
            V_normalized = normalize_vector(V)
            V_p_normalized = normalize_vector(V_p)
            
            # Compute actual matrix multiplication
            actual = torch.matmul(X_normalized, V_normalized.T) * v_norm.T * x_norm
            
            # Forward pass through approximator
            approx = self.approximator(V_normalized, V_p_normalized, X_normalized) * v_norm.T * x_norm
            
            # Compute loss
            loss = torch.mean((actual - approx) ** 2)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'theta': f'{self.approximator.theta.item()*180/3.14159:.1f}°',
                'bias': f'{self.approximator.bias.item():.2f}'
                })
    
    def forward(self, x):
        batch_size = x.size(0)
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Normalize vectors
        x = normalize_vector(x)

        X_T = torch.nn.functional.linear(x, self.T)  # [batch_size, out_features]

        
        # Get signs for all outputs at once
        sign = torch.sign(X_T * 100).detach()  # [batch_size, out_features]
        # Compute approximation for all outputs
        if self.out_features <= self.in_features:
            x_sub = x
        else:
            rep = self.out_features//self.in_features + 1
            x_sub = torch.repeat_interleave(x, rep, -1)
        appx = torch.cos(self.theta) * x_sub[..., :self.out_features] 
        appx_adj = sign * self.sub_bias.unsqueeze(0)
        print(appx.shape, appx_adj.shape)

        output = appx + appx_adj
        output = output * self.v_norm * x_norm
        
        if self.bias is not None:
            output += self.bias
            
        return output

def test_qlinear_layer():
    # Create a small QLinear layer
    in_features = 100
    out_features = 40
    layer = torch.nn.Linear(in_features, out_features)
    layer = QuantizedLinear(layer, 1000, 128)
    
    # Create sample input
    batch_size = 2
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    output = layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)} but got {output.shape}"
    
    # Test with different batch size
    x2 = torch.randn(5, in_features) 
    output2 = layer(x2)
    assert output2.shape == (5, out_features)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_qlinear_layer()
