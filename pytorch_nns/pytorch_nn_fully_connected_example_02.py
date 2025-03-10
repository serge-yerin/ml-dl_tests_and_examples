import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np


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
    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    train_loss /= num_batches
    correct /= size

    return train_loss, 100*correct   # loss.detach().numpy()


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
    return test_loss, 100*correct






if __name__ == '__main__':

    no_epochs = 20
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64
    learning_rate = 1e-4  # 1e-3

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",]


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

    # # Helper function for inline image display # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    # def matplotlib_imshow(img, one_channel=False):
    #     if one_channel:
    #         img = img.mean(dim=0)
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     if one_channel:
    #         plt.imshow(npimg, cmap="Greys")
    #     else:
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # dataiter = iter(train_dataloader)
    # images, labels = next(dataiter)

    # # Create a grid from the images and show them
    # img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=True)
    # print('  '.join(classes[labels[j]] for j in range(4)))

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Using {device} device")

    model = NeuralNetwork().to(device)
    print('\n', model, '\n')

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for t in range(no_epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        loss, acc = train(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(loss)
        train_accuracies.append(acc)

        loss, acc = test(test_dataloader, model, loss_fn)
        val_losses.append(loss)
        val_accuracies.append(acc)

    print("\n Training is finished!")

    # Plotting the Results
    plt.figure(figsize=(14, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(val_losses, label='Validation loss')
    plt.plot(train_losses, label='Train loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.plot(train_accuracies, label='Train accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


    torch.save(model.state_dict(), "model.pth")
    print("\n PyTorch Model State saved to model.pth")

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    
    model.eval()

    x, y = test_data[0][0], test_data[0][1]  # Image and its label

    print('\n Test sample image size:   ', x.size())
    print('\n Test sample class number: ', y)

    with torch.no_grad():
        pred = model(x)  # logits on the output 
        
        class_num = np.squeeze(pred[0].argmax(0).numpy(), axis=0)
        predicted, actual = classes[class_num], classes[y]
        probabilities = (torch.softmax(pred, dim=1)).numpy()
        probabilities = np.squeeze(probabilities, axis=0)
        pred_class_prob = probabilities[class_num]*100

        print('\n Predicted class number: ', class_num)

        print('\n Predicted: ' + predicted + '    Actual: ' + actual + '    Probability: ' + \
              str(np.round(pred_class_prob, 2)) + ' % ')
        
        
        print('\n Tensor of predicted logits: \n', pred)
        
        print('\n Array of predicted probabilities: \n', probabilities)

        print('\nSum of classes probabilities equals: ', np.sum(probabilities), '\n')
        
        

        