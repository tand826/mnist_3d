import argparse
import random

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D


def load_imgs(root, phase, count):
    imgs = []
    labels = []
    paths_all = list(root.glob(f"{phase}/*/*.png"))
    paths = random.sample(paths_all, count)
    for path in paths:
        imgs.append(plt.imread(str(path)).reshape(1, 784))
        labels.append(np.array(int(path.parent.stem)).reshape(1,))

    imgs_reshaped = np.concatenate(imgs, axis=0)
    labels_reshaped = np.concatenate(labels, axis=0)
    return imgs_reshaped, labels_reshaped


def reduce_dim(imgs, svd_dim, out_dim):
    imgs_svd = TruncatedSVD(n_components=svd_dim).fit_transform(imgs)
    imgs_reduced = TSNE(n_components=out_dim,  verbose=1).fit_transform(imgs_svd)
    return imgs_reduced


def show_count(labels):
    for i in range(10):
        print(f"{i} : {np.sum(labels == i)}")


def plot(imgs, labels, out_dim):
    fig = plt.figure(dpi=200)
    if out_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter3D(imgs[:, 0], imgs[:, 1], imgs[:, 2], c=labels)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(imgs[:, 0], imgs[:, 1], c=labels)
    plt.colorbar(scatter)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sn", "--sample_number", type=int, default=100)
    parser.add_argument("-p", "--phase", choices=["train", "test"], default="train")
    parser.add_argument("-sd", "--svd_dimension", type=int, default=100)
    parser.add_argument("-od", "--out_dimension", choices=[2, 3], type=int, default=3)
    parser.add_argument("-sc", "--sampled_count", action='store_true')
    args = parser.parse_args()

    imgs, labels = load_imgs(Path(), args.phase, args.sample_number)
    imgs_reduced = reduce_dim(imgs, args.svd_dimension, args.out_dimension)
    if args.sampled_count:
        show_count(labels)
    plot(imgs_reduced, labels, args.out_dimension)


if __name__ == '__main__':
    main()
