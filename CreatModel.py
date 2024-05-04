from My_Model import ConvNet
import torch.nn as nn
import torch.utils.data
from data import CustomData
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 64
learning_rate = 0.01


# Load the Data
dataset = CustomData('Resources/driving_log.csv', transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)


# Make The Model
model = ConvNet().to(device)
# Choose the Loss Func
criterion = nn.MSELoss()
# Choose the Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training the Model
n_total_steps = len(training_loader)
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(training_loader):
        images = images.to(device)
        labels = labels.to(device).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

print('Finished Training')

PATH = 'Resources/cnn.pth'
torch.save(model, PATH)





"""""
# for debugging
for i, (images, labels) in enumerate(training_loader):
        print(i)
        print(images)
        print(labels)
        output = model(images)
        print(output)
        break
"""