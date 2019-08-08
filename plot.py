import argparse
import random

from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def get_model(name, out_dim, device):
    net = LeNet(out_dim)
    net.load_state_dict(torch.load(f"misc/lenet.pth"))
    net.eval().to(device)
    return net


def reduce_dim(imgs, mid_dim, out_dim, neural_net, device):
    if neural_net:
        print(f"[Neural Net] Reducing dimension from {imgs[0].shape[1] * imgs[0].shape[2]} to {10}...")
        net = get_model(neural_net, 10, device)
        imgs = torch.tensor(imgs, dtype=torch.float).to(device)
        imgs_out = []
        for img in tqdm(imgs):
            imgs_out.append(net(img.unsqueeze(0)).cpu().detach())
        imgs_mid = torch.cat(imgs_out).numpy()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-sn", "--sample_number", type=int, default=100)
    parser.add_argument("-p", "--phase", choices=["train", "test"], default="train")
    parser.add_argument("-nn", "--neural_net", action='store_true')
    parser.add_argument("-md", "--mid_dimension", type=int, default=100)
    parser.add_argument("-od", "--out_dimension", choices=[2, 3], type=int, default=2)
    parser.add_argument("-cs", "--count-sampled", action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    imgs, labels = load_imgs(Path(), args.phase, args.sample_number)
    imgs_reduced = reduce_dim(imgs, args.mid_dimension, args.out_dimension, args.neural_net, device)
    if args.count_sampled:
        show_count(labels)
    plot(imgs_reduced, labels, args.out_dimension)


if __name__ == '__main__':
    main()
