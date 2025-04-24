import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math

class CTRNN(torch.nn.Module):
    """
    Continuous-Time Recurrent Neural Network (CTRNN) implementation.
    
    This implementation follows the standard CTRNN equations:
    τᵢ(dvᵢ/dt) = -vᵢ + Σⱼ wᵢⱼyⱼ + Σₖ wᵢₖuₖ + bᵢ
    yᵢ = σ(vᵢ)
    
    where:
    - vᵢ is the membrane potential of neuron i
    - τᵢ is the time constant of neuron i
    - wᵢⱼ is the weight from neuron j to neuron i
    - yⱼ is the activation of neuron j
    - uₖ is the k-th input
    - bᵢ is the bias of neuron i
    - σ is the activation function (typically tanh or sigmoid)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_sizes: List[int],
                 output_dim: int, 
                 dt: float = 0.1, 
                 tau: float = 1.0,
                 activation: str = 'tanh'):
        """
        Initialize the CTRNN.
        
        Args:
            input_dim: Number of input dimensions
            hidden_sizes: List of hidden layer sizes
            output_dim: Number of output dimensions
            dt: Integration time step (default: 0.1)
            tau: Time constant (default: 1.0)
            activation: Activation function ('tanh' or 'sigmoid')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.dt = dt
        self.tau = tau
        self.alpha = dt / tau
        
        # Set activation function
        self.activation = torch.tanh if activation == 'tanh' else torch.sigmoid
        
        # Build network architecture
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # Output layer
        self.readout = torch.nn.Linear(hidden_sizes[-1], output_dim)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
        
        # Noise parameters (can be adjusted)
        self.membrane_noise = 0.01
        self.activation_noise = 0.01

    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.readout.weight)
        torch.nn.init.zeros_(self.readout.bias)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CTRNN.
        
        Args:
            inputs: Input tensor of shape (seq_len, batch, input_dim)
            
        Returns:
            tuple: (outputs, hidden_states)
                - outputs: Network outputs
                - hidden_states: Hidden layer states
        """
        batch_size = inputs.size(1)
        
        # Initialize membrane potentials
        v = [torch.zeros((batch_size, size), device=inputs.device) 
             for size in self.hidden_sizes]
        
        # Initialize activations
        h = [self.activation(v_i) for v_i in v]
        
        hidden_states = []
        outputs = []
        
        # Process input sequence
        for t in range(inputs.size(0)):
            # Current input
            x = inputs[t]
            
            # Update each layer
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # Input layer
                    current_input = x
                else:
                    # Hidden layer - use previous layer's activation
                    current_input = h[i-1]
                
                # Compute layer input
                layer_input = layer(current_input)
                
                # Update membrane potential (CTRNN equation)
                v[i] = (1 - self.alpha) * v[i] + self.alpha * layer_input
                
                # Add membrane noise
                if self.membrane_noise > 0:
                    v[i] += torch.randn_like(v[i]) * self.membrane_noise
                
                # Compute activation
                h[i] = self.activation(v[i])
                
                # Add activation noise
                if self.activation_noise > 0:
                    h[i] += torch.randn_like(h[i]) * self.activation_noise
            
            # Store hidden states
            hidden_states.append(h[-1])
            
            # Compute output
            output = self.readout(h[-1])
            outputs.append(output)
        
        return (torch.stack(outputs), 
                torch.stack(hidden_states))

    def analyze_stability(self, 
                         x_range: Tuple[float, float] = (-5, 5),
                         y_range: Tuple[float, float] = (-5, 5),
                         n_points: int = 20) -> None:
        """
        Analyze and visualize the stability of the network using phase plane analysis.
        
        Args:
            x_range: Range for x-axis
            y_range: Range for y-axis
            n_points: Number of points in each dimension
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Convert to torch tensors
        states = torch.tensor(np.stack([X, Y], axis=0), dtype=torch.float32)
        
        # Compute derivatives
        with torch.no_grad():
            dx = torch.zeros_like(states[0])
            dy = torch.zeros_like(states[1])
            
            for i in range(n_points):
                for j in range(n_points):
                    state = states[:, i, j]
                    v = self.layers[0](state.unsqueeze(0))
                    h = self.activation(v)
                    output = self.readout(h)
                    
                    dx[i, j] = output[0, 0].item()
                    dy[i, j] = output[0, 1].item()
        
        # Plot phase plane
        plt.figure(figsize=(10, 8))
        plt.quiver(X, Y, dx.numpy(), dy.numpy(), scale=50)
        plt.title('Phase Plane Analysis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()

def train_sine_prediction():
    """Train the CTRNN on sine wave prediction task."""
    # Generate sine wave data
    t = np.linspace(0, 10, 1000)
    inputs = np.sin(t)
    inputs = torch.from_numpy(inputs).float().unsqueeze(1).unsqueeze(1)
    targets = inputs[1:]
    inputs = inputs[:-1]
    
    # Create model
    model = CTRNN(input_dim=1, 
                  hidden_sizes=[32, 32],  # Two hidden layers
                  output_dim=1,
                  dt=0.1,
                  tau=1.0)
    
    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    n_epochs = 500
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6f}')
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Training loss
    plt.subplot(121)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    
    # Predictions
    plt.subplot(122)
    with torch.no_grad():
        predictions, _ = model(inputs)
    plt.plot(t[:-1], inputs.squeeze().numpy(), label='Input')
    plt.plot(t[1:], targets.squeeze().numpy(), label='Target')
    plt.plot(t[1:], predictions.squeeze().numpy(), '--', label='Prediction')
    plt.title('Sine Wave Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Analyze stability
    model.analyze_stability()

if __name__ == "__main__":
    train_sine_prediction()
