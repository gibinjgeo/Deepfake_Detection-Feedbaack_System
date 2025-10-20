import torch
import torch.nn as nn
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dummy model
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
).to(device)

opt = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

# Random data
x = torch.randn(4096, 1024, device=device)
y = torch.randn(4096, 1024, device=device)

# Training loop
for epoch in range(3):
    t0 = time.time()
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} | Loss {loss.item():.4f} | Time {time.time()-t0:.3f}s")
