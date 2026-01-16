from src.dataloader.datasets import MNISTDataset
from src.dataloader.transformation import ChainTransformation, RandomTranslationWithPadding, Flatten
from src.utils.image_classification import plot_images_with_colored_labels
from src.utils.utils import set_seed

set_seed(42)
image_size = 40
data = MNISTDataset(
    train=True,
    batch_size=81,
    shuffle=True,
    drop_last=True,
    transformation=ChainTransformation(
        RandomTranslationWithPadding((image_size, image_size)),
    )
)

plot_images_with_colored_labels(*data.__next__())

