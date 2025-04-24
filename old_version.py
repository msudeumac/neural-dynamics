import torch

class RecurrentLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, dt=10, tau=10):
        super().__init__()
        self.alpha = dt / tau
        self.preact_noise, self.postact_noise = 0.1, 0.1
        self.activation = torch.relu
        self.hidden_size = hidden_size
        self.input_layer = torch.nn.Linear(input_dim, hidden_size)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size)
        ])

        self.input_layer.weight.data.normal_(mean=0.0, std=0.1)
        self.input_layer.bias.data.zero_()
        for layer in self.hidden_layers:
            layer.weight.data.normal_(mean=0.0, std=0.1)
            layer.bias.data.zero_()

    def recurrence(self, fr_t, v_t, u_t):
        """ Recurrence function """
        # through input layer
        w_in_u_t = self.input_layer(u_t)  # u_t @ W_in
        # through hidden layers
        for layer in self.hidden_layers:
            w_hid_fr_t = layer(fr_t)  # fr_t @ W_hid + b
            # update hidden state
            v_t = (1 - self.alpha) * v_t + self.alpha * (w_hid_fr_t + w_in_u_t)

            # add pre-activation noise
            if self.preact_noise > 0:
                preact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.preact_noise
                v_t = v_t + self.alpha * preact_epsilon

            # apply activation function
            fr_t = self.activation(v_t)

            # add post-activation noise
            if self.postact_noise > 0:
                postact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.postact_noise
                fr_t = fr_t + postact_epsilon

        return fr_t, v_t

    def ode_system(self, state, t):
        """ ODE system """
        x, y = state[:, 0], state[:, 1]
        dx_dt = x + y
        dy_dt = -x + y
        return torch.stack([dx_dt, dy_dt], dim=1)

    def forward(self, input):
        """
        Propagate input through the network.
        @param input: shape=(seq_len, batch, input_dim), network input
        @return stacked_states: shape=(seq_len, batch, hidden_size), stack of hidden layer status
        """
        #v_t is the batch size. This variable is used to initialize the hidden state v_t in the forward method of the RecurrentLayer class. The size of v_t is (input.size(1), self.hidden_size), where input.size(1) corresponds to the batch size.
        v_t = torch.zeros((input.size(1), self.hidden_size), device=input.device)
        fr_t = self.activation(v_t)
        # update hidden state and append to stacked_states
        stacked_states = []
        for i in range(input.size(0)):
            fr_t, v_t = self.recurrence(fr_t, v_t, input[i])
            # append to stacked_states
            stacked_states.append(fr_t)

        return torch.stack(stacked_states, dim=0)

class CTRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dt=10, tau=100):
        super().__init__()
        self.recurrent_layer = RecurrentLayer(input_dim, hidden_size, dt=dt, tau=tau)
        self.readout_layer = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, inputs):
        hidden_states = self.recurrent_layer(inputs)
        output = self.readout_layer(hidden_states.float())
        return output, hidden_states

import numpy as np
import matplotlib.pyplot as plt

# predict sin wave
inputs = np.sin(np.linspace(0, 10, 1000))
inputs = torch.from_numpy(inputs).float().unsqueeze(1).unsqueeze(1)
labels = inputs[1:]
inputs = inputs[:-1]

plt.plot(inputs.squeeze(1).squeeze(1).numpy())
plt.plot(labels.squeeze(1).squeeze(1).numpy())
plt.xlabel("Time")
plt.ylabel("Inputs")
plt.show()


ctrnn = CTRNN(input_dim=1, hidden_size=10, output_dim=1)
optimizer = torch.optim.Adam(ctrnn.parameters(), lr=0.001)


losses = []
for epoch in range(500):
    outputs, states = ctrnn(inputs)
    loss = torch.nn.MSELoss()(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')

epochs = np.arange(len(losses))
plt.plot(epochs, losses)
plt.title("CTRNN Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()








import numpy as np
import matplotlib.pyplot as plt

NI, NJ = 15, 15
X, Y = np.meshgrid(np.linspace(-5, 5, NI), np.linspace(-5, 5, NJ))
t = 0
u = np.zeros_like(X)
v = np.zeros_like(Y)

def ode_states(state, t):
    """ ODE system """
    grid1, grid2 = state[0], state[1]
    dx_dt = grid1 + grid2
    dy_dt = -grid1 + grid2
    return np.array([dx_dt, dy_dt])

for i in range(NI):
    for j in range(NJ):
        grid1 = X[i, j]
        grid2 = Y[i, j]
        yprime = ode_states(np.array([grid1, grid2]), t)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

QuiverPlot = plt.quiver(X, Y, u, v, color='blue')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title("Phase Plane")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

