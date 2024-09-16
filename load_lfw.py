from collections import Counter
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image


TARGET_HEIGHT = 224
TARGET_WIDTH = 224


def _resize_image(image):
    """Crops the image to be 224x224 - the minimum out of the three models we are testing.

    Args:
        image: 3D numpy array representing an image in RGB format

    Returns:
        the resized image as a numpy array.
    """
    image_tensor = tf.convert_to_tensor(image)
    cropped_image = tf.image.resize_with_crop_or_pad(
        image_tensor, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH
    )

    return cropped_image.numpy()


def _get_top_n_classes(lfw_root_dir, n=10):
    """Gets the top n classes in the LFW data set dynamically.

    Args:
        lfw_root_dir: path to the root of the LFW dataset directory
        n: Number of classes to retrieve. Defaults to 10.
    """
    # get all nested directories (class labels)
    all_classes = [d.name for d in Path(lfw_root_dir).iterdir() if d.is_dir()]

    # count number of samples per class
    class_counts = Counter()
    for class_name in all_classes:
        class_dir = Path(lfw_root_dir) / class_name
        num_images = len(list(class_dir.glob("*.jpg")))
        class_counts[class_name] = num_images

    # Return top n classes
    return [cls for cls, _ in class_counts.most_common(n)]


def load_lfw_dataset(lfw_root_dir, n=10):
    """Loads the Labelled Faces in the Wild dataset.

    Args:
        lfw_root_dir: path to the root of the LFW dataset directory
        n: Number of classes to retrieve. Defaults to 10.

    Returns:
        DataFrame: dataframe of image data and corresponding label.
    """
    top_classes = _get_top_n_classes(lfw_root_dir, n)
    encoding_map = {top_classes[i]: i for i in range(len(top_classes))}

    X = []
    y = []

    for class_name in top_classes:
        print(f"Loading class: {class_name}")
        class_dir = Path(lfw_root_dir) / class_name
        for img_path in class_dir.glob("*.jpg"):
            # Read the image and convert to numpy array
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            resized = _resize_image(
                img_array
            )  # Resize to 224x224 - minimum image requirement by the models under test.

            X.append(resized)
            y.append(encoding_map[class_name])

    return np.array(X), np.array(y), top_classes
