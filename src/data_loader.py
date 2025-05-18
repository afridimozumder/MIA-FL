import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_indices, val_indices = train_test_split(range(len(trainset)), test_size=0.2, random_state=42)
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)
    batch_size = 64
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader

def load_svhn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    train_indices, val_indices = train_test_split(range(len(trainset)), test_size=0.2, random_state=42)
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)
    batch_size = 64
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader
