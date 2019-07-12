from pathlib import Path
from PIL import Image
from misc.model import LeNet
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class MnistDataset(Dataset):
    """Mnist dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.paths = list(self.root_dir.glob("**/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        image = Image.open(img_name)
        label = torch.tensor(int(img_name.parent.stem))

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    class Args():
        def __init__(self):
            pass

    args = Args()
    args.batch_size = 64
    args.test_batch_size = 1000
    args.epochs = 10
    args.lr = 0.01
    args.momentum = 0.5
    args.no_cuda = False
    args.seed = 1
    args.log_interval = 10
    args.save_model = True

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MnistDataset(root_dir="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = MnistDataset(root_dir="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=4)

    model = LeNet().to(device)
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == '__main__':
    main()
