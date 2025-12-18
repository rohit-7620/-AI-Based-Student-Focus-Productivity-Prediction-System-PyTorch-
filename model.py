import torch
import torch.nn as nn
import numpy as np

class FocusPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

# Train model
model = FocusPredictionModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = np.array([
    [8,2,5,4,6],
    [5,5,7,2,3],
    [3,7,8,1,2],
    [10,1,4,6,7],
    [6,4,6,3,4],
    [4,6,7,2,3],
    [9,2,5,5,6],
    [2,8,8,1,1]
], dtype=np.float32)

y = np.array([0,1,2,0,1,2,0,2])

X = torch.tensor(X)
y = torch.tensor(y)

for _ in range(300):
    out = model(X)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict_focus(data):
    data = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        result = model(data)
        return torch.argmax(result).item()
