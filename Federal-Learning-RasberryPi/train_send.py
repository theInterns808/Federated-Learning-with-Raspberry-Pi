import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import asyncio
import websockets

# Define your model (example)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load and normalize the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(2):  # number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')

# Save the model weights
torch.save(model.state_dict(), "mnist_model_weights.pth")

# Function to send the model weights to the server in chunks
async def send_weights():
    uri = "ws://192.168.8.160:8765"  # Replace '8765' with the actual port number
    async with websockets.connect(uri) as websocket:
        # Open the file in binary mode
        with open("mnist_model_weights.pth", "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # Read file in chunks of 1 MB
                if not chunk:
                    break
                await websocket.send(chunk)
                print("Sent a chunk to the server")
        await websocket.send("EOF")  # Send an EOF signal
        print("All chunks sent to the server")

# Run the asyncio event loop to send the weights
asyncio.get_event_loop().run_until_complete(send_weights())
