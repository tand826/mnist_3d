import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D


def load_imgs(root, phase, count):
    imgs = []
    labels = []
    paths = list(root.glob(f"{phase}/*/*.png"))[:count]
    for path in paths:
        imgs.append(plt.imread(str(path)).reshape(1, 784))
        labels.append(np.array(int(path.parent.stem)).reshape(1,))
    return imgs, labels


def reduce_dim(imgs, labels, dimension):
    imgs_reshaped = np.concatenate(imgs, axis=0)
    print(imgs_reshaped.shape)
    labels_reshaped = np.concatenate(labels, axis=0)
    print(labels_reshaped.shape)

    svd = TruncatedSVD(n_components=dimension)
    imgs_reduced_ = svd.fit_transform(imgs_reshaped)

    imgs_reduced = TSNE(n_components=3,  verbose=1).fit_transform(imgs_reduced_)
    print(imgs_reduced.shape)

    return imgs_reduced, labels_reshaped


def plot(imgs, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(imgs.shape, labels.shape)
    ax.scatter3D(imgs[:, 0], imgs[:, 1], imgs[:, 2], c=labels)
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    parser.add_argument("phase", choices=["train", "test"])
    parser.add_argument("dimension", type=int)
    args = parser.parse_args()

    imgs, labels = load_imgs(Path(), args.phase, args.count)
    imgs_reduced, labels = reduce_dim(imgs, labels, args.dimension)
    plot(imgs_reduced, labels)


if __name__ == '__main__':
    main()
