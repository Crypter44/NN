from src.dataloader.datasets import MNISTDataset
from src.dataloader.transformation import ChainTransformation, Normalize, RandomTranslationWithPadding, Flatten
from src.utils.image_classification import plot_images_with_colored_labels

norm = Normalize()

data = MNISTDataset(
    train=False,
    batch_size=81,
    shuffle=False,
    drop_last=False,
    transformation=ChainTransformation(
        # norm,
        RandomTranslationWithPadding((40, 40)),
        # Flatten()
    )
)

images, labels = next(iter(data))
plot_images_with_colored_labels(
    images.reshape(-1, 40, 40),
    labels
)
