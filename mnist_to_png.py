import os
from PIL import Image
import chainer


def save(data, index, num, phase):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    img2 = img.resize((28, 28))
    filename = phase + "/" + str(num) + "/" + "{0:04d}".format(index) + ".png"
    img2.save(filename)
    print(filename)


def main():
    train, test = chainer.datasets.get_mnist()
    for phase, data in {"train": train, "test": test}.items():
        if os.path.isdir(phase) is False:
            os.mkdir(phase)
        for i in range(10):
            dirname = phase + "/" + str(i)
            if os.path.isdir(dirname) is False:
                os.mkdir(dirname)
        for i in range(len(data)):
            save(data[i][0], i, data[i][1], phase)


if __name__ == '__main__':
    main()
