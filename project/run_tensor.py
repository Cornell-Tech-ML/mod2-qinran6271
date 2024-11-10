"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# # TODO: Implement for Task 2.5.

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        
        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)    

    def forward(self, x):

        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size
    
    def forward(self, x):
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)


# class Network(minitorch.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         # Input layer to hidden layer 1
#         self.layer1 = RParam(2, hidden_size)  # 2 -> hidden_size
#         self.bias1 = RParam(hidden_size)

#         # Hidden layer 1 to hidden layer 2
#         self.layer2 = RParam(hidden_size, hidden_size)  # hidden_size -> hidden_size
#         self.bias2 = RParam(hidden_size)

#         # Hidden layer 2 to output
#         self.layer3 = RParam(hidden_size, 1)  # hidden_size -> 1
#         self.bias3 = RParam(1)

#     def forward(self, x):
#         batch_size, input_features = x.shape[0], x.shape[1]
#         print("layer:", x)

#         # First layer
#         x = x.view(batch_size, input_features, 1)  # Adjust x to shape (batch_size, input_features, 1)
#         weights = self.layer1.value  # Shape (input_features, hidden_size)
#         x = x * weights # Shape (batch_size, input_features, hidden_size)
#         print("After weight:", x)
#         x = x.sum(1)   # Sum along the 1st dimension (input_features) Shape(batch_size, 1, hidden_size)
#         print("After sum:", x.shape)
#         x += self.bias1.value # Shape(batch_size, 1, hidden_size)
#         batch_size, input_features = x.shape[0], x.shape[-1]
#         x = x.view(batch_size, input_features)  # Adjust to (batch_size, hidden_size)
#         print("After first layer:", x.shape)
#         x = x.relu()

#         # Second layer
#         x = x.view(batch_size, input_features, 1)  # Adjust x to shape (batch_size, input_features, 1)
#         weights = self.layer2.value  # Shape (input_features, hidden_size)
#         x = x * weights # Shape (batch_size, input_features, hidden_size)
#         x = x.sum(1) # Sum along the 1st dimension (hidden_size) Shape(batch_size, 1, hidden_size)
#         x += self.bias2.value #Shape(batch_size, 1, hidden_size)
#         batch_size, input_features = x.shape[0], x.shape[-1]
#         x = x.view(batch_size, input_features)  # Adjust to (batch_size, hidden_size)
#         print("After second layer:", x.shape)
#         x = x.relu()

#         # #Output layer
#         x = x.view(batch_size, input_features, 1)  # Adjust x to shape (batch_size, input_features, 1)
#         weights = self.layer3.value  # Shape (input_features, 1)
#         x = (x * weights) # Shape (batch_size, input_features, hidden_size)
#         x = x.sum(1) # Shape (batch_size, 1, hidden_size)
#         x +=  self.bias3.value # Shape (batch_size, 1, hidden_size)
#         batch_size, input_features = x.shape[0], x.shape[-1]
#         x = x.view(batch_size, input_features)  # Adjust to (batch_size, hidden_size)
#         print("After output layer:", x.shape)
#         x = x.sigmoid()

#         return x


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
