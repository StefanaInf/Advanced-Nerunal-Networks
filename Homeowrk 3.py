import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import timm
from preact_resnet import PreActResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet18CIFAR10(nn.Module):
    def __init__(self):
        super(ResNet18CIFAR10, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=10)

    def forward(self, x):
        return self.model(x)

class CachedDataset(Dataset):
    def __init__(self, dataset_name, transform=None, is_training=True, cache_size=1000):
        self.dataset_name = dataset_name
        self.transform = transform
        self.cache_size = cache_size

        if dataset_name == "MNIST":
            self.dataset = datasets.MNIST(root='./data', train=is_training, download=True)
        elif dataset_name == "CIFAR10":
            self.dataset = datasets.CIFAR10(root='./data', train=is_training, download=True)
        elif dataset_name == "CIFAR100":
            self.dataset = datasets.CIFAR100(root='./data', train=is_training, download=True)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        self.cache = {}
        self.cache_keys = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index in self.cache:
            image, target = self.cache[index]
        else:
            image, target = self.dataset[index]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (copy.deepcopy(image), target)
                self.cache_keys.append(index)
            else:
                oldest_index = self.cache_keys.pop(0)
                del self.cache[oldest_index]
                self.cache[index] = (copy.deepcopy(image), target)
                self.cache_keys.append(index)

        if self.transform:
            image = self.transform(image)

        return image, target

class ConfigurableDataLoader:
    def __init__(self, dataset_name, batch_size, is_training, augmentations=None, model_name=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.is_training = is_training
        self.model_name = model_name

        if self.is_training:
            self.augmentations = augmentations if augmentations else transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        else:
            self.augmentations = augmentations if augmentations else transforms.Compose([
                transforms.ToTensor(),
            ])

        self.dataset = CachedDataset(self.dataset_name, transform=self.augmentations, is_training=self.is_training)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.is_training)

        self.model = self.load_model()

    def load_model(self):
        if self.dataset_name == "CIFAR10":
            if self.model_name == "resnet18":
                model = timm.create_model('resnet18', pretrained=True, num_classes=10)
            elif self.model_name == "PreActResNet18":
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
            else:
                raise ValueError(f"Model {self.model_name} is not supported for CIFAR10.")
        elif self.dataset_name == "CIFAR100":
            if self.model_name == "resnet18":
                model = timm.create_model('resnet18', pretrained=True, num_classes=100)
            elif self.model_name == "PreActResNet18":
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 100)
            else:
                raise ValueError(f"Model {self.model_name} is not supported for CIFAR100.")
        elif self.dataset_name == "MNIST":
            if self.model_name == "MLP":
                model = MLP()
            elif self.model_name == "LeNet":
                model = LeNet()
            else:
                raise ValueError(f"Model {self.model_name} is not supported for MNIST.")
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

        return model

    def load_optimizer(self, model, optimizer_name, learning_rate=0.001, momentum=0.9, weight_decay=0.0):
        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD_momentum":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "SGD_nesterov":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True,
                                  weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "RMSProp":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} is not supported.")

        return optimizer

    def create_lr_scheduler(self, optimizer, scheduler_name, **kwargs):
        if scheduler_name == "StepLR":
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "ReduceLROnPlateau":
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        else:
            raise ValueError(f"Scheduler {scheduler_name} is not supported.")

        return scheduler

    def get_loader(self):
        return self.data_loader


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    dataset_name = "CIFAR100"
    model_name = "resnet18"
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 100
    early_stopping_patience = 10

    train_loader = ConfigurableDataLoader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        is_training=True,
        model_name=model_name,
    )

    validation_loader = ConfigurableDataLoader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        is_training=False,
        model_name=model_name,
    )

    model = train_loader.load_model().to(device)

    optimizer = train_loader.load_optimizer(model, optimizer_name="SGD_momentum", learning_rate=learning_rate, momentum=momentum)

    scheduler = train_loader.create_lr_scheduler(optimizer, scheduler_name="StepLR", step_size=30, gamma=0.1)

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader.get_loader()):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.get_loader())
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_data, val_target in validation_loader.get_loader():
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_loss += torch.nn.functional.cross_entropy(val_output, val_target).item()

        val_loss /= len(validation_loader.get_loader())
        print(f"Validation Loss: {val_loss:.4f}")
        scheduler.step()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    print("Testing the model...")
    test_loader = validation_loader.get_loader()
    criterion = nn.CrossEntropyLoss()
    test_model(model, test_loader, criterion, device)
