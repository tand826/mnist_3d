import argparse
import random

from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D

from misc.model import LeNet


def load_imgs(root, phase, count):
    imgs = []
    labels = []
    paths_all = list(root.glob(f"{phase}/*/*.png"))
    paths = random.sample(paths_all, count)
    for path in paths:
        imgs.append(plt.imread(str(path)).reshape(1, 28, 28))
        labels.append(np.array(int(path.parent.stem)).reshape(1,))

    labels_reshaped = np.concatenate(labels, axis=0)
    return imgs, labels_reshaped


def transform_imgs(imgs):
    return np.concatenate(imgs, axis=0).reshape(len(imgs), -1)


def get_model(name, out_features, device):

    if name == "lenet":
        net = LeNet()
        net.load_state_dict(torch.load("misc/normal.pth"))
    elif name == "resnet18":
        net = models.resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(512, out_features)
    elif name == "resnet152":
        net = models.resnet152(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(2048, out_features)
    elif name == "inception_v3":
        net = models.inception_v3(pretrained=True)
        net.fc = nn.Linear(2048, out_features)
    elif name == "densenet121":
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, out_features)
    elif name == "densenet201":
        net = models.densenet201(pretrained=True)
        net.classifier = nn.Linear(1920, out_features)

    net.eval().to(device)
    return net


def reduce_dim(imgs, mid_dim, out_dim, nn, device):
    if nn:
        print(f"[Neural Net] Reducing dimension from {imgs[0].shape[1] * imgs[0].shape[2]} to {mid_dim}...")
        net = get_model(nn, mid_dim, device)
        imgs = torch.tensor(imgs, dtype=torch.float).to(device)
        imgs_mid = []
        for img in tqdm(imgs):
            imgs_mid.append(net(img.unsqueeze(0)).cpu().detach())
        imgs_mid = torch.cat(imgs_mid).numpy()
    else:
        imgs_transformed = transform_imgs(imgs)
        imgs_mid = TruncatedSVD(n_components=mid_dim).fit_transform(imgs_transformed)
    imgs_reduced = TSNE(n_components=out_dim,  verbose=1).fit_transform(imgs_mid)
    return imgs_reduced


def show_count(labels):
    for i in range(10):
        print(f"{i} : {np.sum(labels == i)}")


def plot(imgs, labels, out_dim):
    fig = plt.figure(dpi=200)
    if out_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter3D(imgs[:, 0], imgs[:, 1], imgs[:, 2], c=labels, linewidths=0.3)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(imgs[:, 0], imgs[:, 1], c=labels, linewidths=0.3)
    plt.colorbar(scatter)
    plt.show()


def main():
    nn_choices = ["lenet", "resnet18", "resnet152", "inception_v3", "densenet121", "densenet201"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-sn", "--sample_number", type=int, default=100)
    parser.add_argument("-p", "--phase", choices=["train", "test"], default="train")
    parser.add_argument("-n", "--normalization", action="store_true")
    parser.add_argument("-nn", "--neural_net", choices=nn_choices, default=False)
    parser.add_argument("-md", "--mid_dimension", type=int, default=100)
    parser.add_argument("-od", "--out_dimension", choices=[2, 3], type=int, default=3)
    parser.add_argument("-sc", "--sampled_count", action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    imgs, labels = load_imgs(Path(), args.phase, args.sample_number)
    imgs_reduced = reduce_dim(imgs, args.mid_dimension, args.out_dimension, args.neural_net, device)
    if args.sampled_count:
        show_count(labels)
    plot(imgs_reduced, labels, args.out_dimension)


if __name__ == '__main__':
    main()
