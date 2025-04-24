# neural-dynamics
A PyTorch implementation of Continuous-Time Recurrent Neural Networks (CTRNNs) for modeling dynamical systems and temporal patterns. Features include flexible architecture, multiple activation functions, built-in noise parameters, and phase plane analysis for stability visualization.

# Continuous-Time Recurrent Neural Network (CTRNN) Implementation

This repository contains implementations of Continuous-Time Recurrent Neural Networks (CTRNNs) in PyTorch. CTRNNs are a type of neural network that operate in continuous time, making them particularly suitable for modeling dynamical systems and temporal patterns.

## Features

- Flexible CTRNN architecture with configurable hidden layers
- Support for different activation functions (tanh, sigmoid)
- Built-in noise parameters for robust learning
- Phase plane analysis for stability visualization
- Time series prediction capabilities
- Comprehensive documentation and type hints

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ctrnn-implementation.git
cd ctrnn-implementation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic CTRNN Implementation

```python
from ctrnn import CTRNN

# Initialize the network
model = CTRNN(
    input_dim=1,
    hidden_sizes=[32, 32],
    output_dim=1,
    dt=0.1,
    tau=1.0,
    activation='tanh'
)

# Forward pass
outputs, hidden_states = model(inputs)
```

### Training Example

```python
# Train on sine wave prediction
model.train_sine_prediction()
```

### Stability Analysis

```python
# Analyze network stability
model.analyze_stability(
    x_range=(-5, 5),
    y_range=(-5, 5),
    n_points=20
)
```

## Architecture

The CTRNN implementation follows the standard equations:

```
τᵢ(dvᵢ/dt) = -vᵢ + Σⱼ wᵢⱼyⱼ + Σₖ wᵢₖuₖ + bᵢ
yᵢ = σ(vᵢ)
```

Where:
- vᵢ: membrane potential of neuron i
- τᵢ: time constant of neuron i
- wᵢⱼ: weight from neuron j to neuron i
- yⱼ: activation of neuron j
- uₖ: k-th input
- bᵢ: bias of neuron i
- σ: activation function

## Applications

- Time series prediction
- Dynamical systems modeling
- Robotics control
- Pattern recognition in continuous-time signals
- Neuroscience research

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by biological neural networks
- Based on continuous-time recurrent neural network theory
- Built with PyTorch for efficient computation

## Contact

For questions or suggestions, please open an issue or contact the maintainer.
