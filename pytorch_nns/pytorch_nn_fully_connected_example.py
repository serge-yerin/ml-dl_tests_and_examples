import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt



# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.Dropout(p=0.7),  # Dropout layer with 50% probability
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # probability of an element to be zeroed.
        self.layer2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # probability of an element to be zeroed.
        self.layer3 = nn.Linear(512, 10)        
 
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.act1(self.layer1(x)))
        x = self.dropout2(self.act2(self.layer2(x)))
        x = self.layer3(x)
        return x




def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"\n Test Error: \n Accuracy: {(100 * correct):>0.1f}%,    Avg loss: {test_loss:>8f} \n")



if __name__ == '__main__':

    no_epochs = 8
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64
    learning_rate = 1e-3


    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(),)

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(),)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Using {device} device")


    model = NeuralNetwork().to(device)
    print('\n', model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for t in range(no_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("\n Training is finished!")

    torch.save(model.state_dict(), "model.pth")
    print("\n PyTorch Model State saved to model.pth")

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",]

    model.eval()

    x, y = test_data[0][0], test_data[0][1]

    print('\n Image size:   ', x.size())
    print('\n Class number: ', y)

    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]

        print(pred[0].argmax(0))
        
        print(pred)

        print(f'Predicted: "{predicted}", Actual: "{actual}"')