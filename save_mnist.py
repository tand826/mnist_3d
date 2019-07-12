from torchvision import datasets, transforms
from pathlib import Path


def main():
    to_pil_image = transforms.ToPILImage()
    for phase in ["train", "test"]:
        base = Path(phase)
        mkdirs(base)
        is_train = phase == "train"
        mnist = datasets.MNIST(".", train=is_train, download=True)
        for idx, (img, label) in enumerate(zip(mnist.data, mnist.targets)):
            name = base/str(label.item())/f"{idx:04}.png"
            to_pil_image(img).save(str(name))


def mkdirs(base):
    if not base.exists():
        base.mkdir(exist_ok=True)
    for i in range(10):
        (base/str(i)).mkdir(exist_ok=True)


if __name__ == '__main__':
    main()
