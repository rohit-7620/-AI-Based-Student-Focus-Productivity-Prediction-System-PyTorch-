import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# 1. Dataset (Synthetic but Realistic)
# -----------------------------
# Features: [ScreenTime, StudyHours, SleepHours, SocialMedia, Breaks]
X = np.array([
    [8, 2, 5, 4, 6],
    [5, 5, 7, 2, 3],
    [3, 7, 8, 1, 2],
    [10, 1, 4, 6, 7],
    [6, 4, 6, 3, 4],
    [4, 6, 7, 2, 3],
    [9, 2, 5, 5, 6],
    [2, 8, 8, 1, 1]
], dtype=np.float32)

# Labels: 0=Low, 1=Medium, 2=High Focus
y = np.array([0, 1, 2, 0, 1, 2, 0, 2])

# Convert to PyTorch tensors
X = torch.tensor(X)
y = torch.tensor(y)

# -----------------------------
# 2. Neural Network Model
# -----------------------------
class FocusPredictionModel(nn.Module):
    def __init__(self):
        super(FocusPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)

model = FocusPredictionModel()

# -----------------------------
# 3. Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 300
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nTraining Completed Successfully!")

# -----------------------------
# 5. Prediction Function
# -----------------------------
def predict_focus(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output).item()

    focus_map = {0: "Low Focus", 1: "Medium Focus", 2: "High Focus"}
    return focus_map[prediction]

# -----------------------------
# 6. Test Prediction
# -----------------------------
sample_student = [[4, 6, 8, 2, 2]]  # New student data
result = predict_focus(sample_student)
print("\nPredicted Focus Level:", result)
