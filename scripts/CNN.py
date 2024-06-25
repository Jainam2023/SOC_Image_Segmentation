# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-02T17:48:34.375877Z","iopub.execute_input":"2024-06-02T17:48:34.376332Z","iopub.status.idle":"2024-06-02T17:48:42.626946Z","shell.execute_reply.started":"2024-06-02T17:48:34.376296Z","shell.execute_reply":"2024-06-02T17:48:42.625635Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(root='./', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-02T17:48:42.628910Z","iopub.execute_input":"2024-06-02T17:48:42.629475Z","iopub.status.idle":"2024-06-02T17:48:42.785868Z","shell.execute_reply.started":"2024-06-02T17:48:42.629439Z","shell.execute_reply":"2024-06-02T17:48:42.784649Z"}}
#test
dataiter=iter(train_loader)
image,label=next(dataiter)
conv1=nn.Conv2d(1,3,5)
conv2 = nn.Conv2d(3, 8, 5)
pool = nn.MaxPool2d(2, 2)
x1=pool(F.relu(conv1(image)))
x2=pool(F.relu(conv2(x1)))
x3=x2.view(-1, 8*4*4)
print(x1.shape, x2.shape, x3.shape)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-02T17:49:52.190544Z","iopub.execute_input":"2024-06-02T17:49:52.190980Z","iopub.status.idle":"2024-06-02T17:49:52.198364Z","shell.execute_reply.started":"2024-06-02T17:49:52.190948Z","shell.execute_reply":"2024-06-02T17:49:52.196986Z"}}
n_classes=10
n_inp_channels=1
ker_size=5
c1_out=3
c2_out=8
fc1_out=70
fc2_out=30
n_classes=10
img_dim=28

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-02T17:49:56.329338Z","iopub.execute_input":"2024-06-02T17:49:56.329819Z","iopub.status.idle":"2024-06-02T17:49:56.342785Z","shell.execute_reply.started":"2024-06-02T17:49:56.329781Z","shell.execute_reply":"2024-06-02T17:49:56.341212Z"}}
class ConvNet(nn.Module):
    def __init__(self, img_dim, n_inp_channels, ker_size, c1_out, c2_out, fc1_out, fc2_out, n_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_inp_channels, c1_out, ker_size)
        dim1=(28-ker_size+1)//2
        print(dim1)
        dim2=(dim1-ker_size+1)//2
        print(dim2)
        self.f_length=dim2*dim2*c2_out
        print(self.f_length)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1_out, c2_out, ker_size)
        self.fc1 = nn.Linear(self.f_length, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, self.f_length)        
        x = F.relu(self.fc1(x))       
        x = F.relu(self.fc2(x))       
        x = self.fc3(x)                 
        return x

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-02T17:50:23.959445Z","iopub.execute_input":"2024-06-02T17:50:23.959884Z","iopub.status.idle":"2024-06-02T17:54:41.307357Z","shell.execute_reply.started":"2024-06-02T17:50:23.959852Z","shell.execute_reply":"2024-06-02T17:54:41.305378Z"}}
model = ConvNet(img_dim,n_inp_channels, ker_size, c1_out, c2_out, fc1_out, fc2_out, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-02T17:55:04.311113Z","iopub.execute_input":"2024-06-02T17:55:04.311661Z","iopub.status.idle":"2024-06-02T17:55:08.096669Z","shell.execute_reply.started":"2024-06-02T17:55:04.311620Z","shell.execute_reply":"2024-06-02T17:55:08.095272Z"}}
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {acc} %')
