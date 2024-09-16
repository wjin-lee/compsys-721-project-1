import os
import joblib
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import ResNet152, EfficientNetB0, VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)
from keras.src.legacy.preprocessing.image import (
    ImageDataGenerator,
)  # Deprecated from TF 2.16 - using legacy compat for now to maintain backwards-compatibility with lower TF versions (remove src.legacy).
from keras import layers, models
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
from load_lfw import load_lfw_dataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import seaborn as sns
from sklearn.preprocessing import label_binarize

SEED = 3
LFW_DIR = "/mnt/c/_UoA/Part-4-2024/compsys721/project-1/compsys-721-project-1/lfw"
EARLY_STOPPING_PATIENCE = 3
MAX_FINE_TUNE_EPOCHS = 50
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings


def build_resnet(learning_rate=0.0001, num_classes=10):
    """Builds a ResNet-152 model pretrained on the ImageNet dataset."""

    orig_model = ResNet152(weights="imagenet")  # For plotting only.
    base_model = ResNet152(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = True  # Enable fine-tuning

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Plot model architectures
    plot_model(orig_model, to_file="resnet_orig_top.png", show_shapes=True)
    # plot_model(base_model, to_file="resnet_base.png", show_shapes=True)
    plot_model(model, to_file="resnet_fine_tuned.png", show_shapes=True)

    return model


def build_efficientnetb0(learning_rate=0.0001, num_classes=10):
    """Builds a EfficientNet-B0 model pretrained on the ImageNet dataset."""

    orig_model = EfficientNetB0(weights="imagenet")  # For plotting only.
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = True  # Enable fine-tuning

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Plot model architectures
    plot_model(orig_model, to_file="efficient_net_b0_orig_top.png", show_shapes=True)
    plot_model(base_model, to_file="efficient_net_b0_base.png", show_shapes=True)
    plot_model(model, to_file="efficient_net_b0_fine_tuned.png", show_shapes=True)

    return model


def build_vgg(learning_rate=0.0001, num_classes=10):
    """Builds a VGG model pretrained on the ImageNet dataset."""

    orig_model = VGG16(weights="imagenet")  # For plotting only.
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Enable fine-tuning

    model = models.Sequential(
        [
            base_model,
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dense(4096, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Plot model architectures
    plot_model(orig_model, to_file="vgg_orig_top.png", show_shapes=True)
    plot_model(base_model, to_file="vgg_base.png", show_shapes=True)
    plot_model(model, to_file="vgg_fine_tuned.png", show_shapes=True)

    return model


def fine_tuned_model(model, training_datagen, val_datagen):
    """Fine tunes a given model with the given training and validation data.

    Args:
        model: A TF model to be fine-tuned.
        training_datagen: A data generator for the training data
        val_datagen: A data generator for the validation data
    """

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Watching validation loss
        patience=EARLY_STOPPING_PATIENCE,  # Number of epochs with no/negative improvement after which training will be stopped
        restore_best_weights=True,
    )

    history = model.fit(
        training_datagen,
        validation_data=val_datagen,
        epochs=MAX_FINE_TUNE_EPOCHS,
        callbacks=[early_stopping_callback],
    )

    return model, history


def get_datagens(
    preprocessing_function, X_train, y_train, X_val, y_val, X_test, y_test, batch_size
):
    """Prepares the data generators for training, validation and test data given a preprocessing function."""

    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_gen = datagen.flow(X_val, y_val, batch_size=batch_size)
    test_gen = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen


def plot_history(history, model_name=""):
    """Plots the history graph from TF's model training history log"""

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{model_name} Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()


def save_to_result_json(entry):
    """Saves a grid search result entry to the results JSON."""
    # Load the existing JSON file
    with open("result.json", "r") as file:
        data = json.load(file)

    # Add new result entry
    data["results"].append(entry)

    # Save the updated list back to the JSON file
    with open("result.json", "w") as file:
        json.dump(data, file, indent=4)


def gridsearch(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """Performs a search for the best hyper parameter configuration"""

    results = []
    for batch_size in [8, 16, 32]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            print(f"BATCH: {batch_size} | LEARN: {learning_rate}")
            # # ========================= EFFICIENT NET B0 =========================
            en_train_gen, en_val_gen, en_test_gen = get_datagens(
                efficientnet_preprocess,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                batch_size,
            )

            # Fine tune
            efficientnet_model, _ = fine_tuned_model(
                build_efficientnetb0(learning_rate=learning_rate),
                en_train_gen,
                en_val_gen,
            )

            # Evaluate
            efficientnet_test_loss, efficientnet_test_acc = efficientnet_model.evaluate(
                en_test_gen
            )
            print(
                f"EfficientNetB0 Test Accuracy: {efficientnet_test_acc * 100:.2f}% | Test Loss: {efficientnet_test_loss}"
            )
            save_to_result_json(
                {
                    "arch": "EfficientNet",
                    "learn_rate": learning_rate,
                    "batch_size": batch_size,
                    "test_acc": efficientnet_test_acc,
                    "test_loss": efficientnet_test_loss,
                }
            )
            # # =====================================================================

            # # =============================== RESNET ==============================
            # rn_train_gen, rn_val_gen, rn_test_gen = get_datagens(
            #     resnet_preprocess,
            #     X_train,
            #     y_train,
            #     X_val,
            #     y_val,
            #     X_test,
            #     y_test,
            #     batch_size,
            # )

            # # Fine tune
            # resnet_model, _ = fine_tuned_model(
            #     build_resnet(learning_rate=learning_rate), rn_train_gen, rn_val_gen
            # )

            # # Evaluate
            # resnet_test_loss, resnet_test_acc = resnet_model.evaluate(rn_test_gen)
            # print(
            #     f"ResNet-152 Test Accuracy: {resnet_test_acc * 100:.2f}% | Test Loss: {resnet_test_loss}"
            # )

            # save_to_result_json(
            #     {
            #         "arch": "ResNet",
            #         "learn_rate": learning_rate,
            #         "batch_size": batch_size,
            #         "test_acc": resnet_test_acc,
            #         "test_loss": resnet_test_loss,
            #     }
            # )
            # # =====================================================================

            # =============================== VGG ==============================
            # vg_train_gen, vg_val_gen, vg_test_gen = get_datagens(
            #     vgg_preprocess,
            #     X_train,
            #     y_train,
            #     X_val,
            #     y_val,
            #     X_test,
            #     y_test,
            #     batch_size,
            # )

            # # Fine tune
            # vgg_model, _ = fine_tuned_model(
            #     build_vgg(learning_rate=learning_rate), vg_train_gen, vg_val_gen
            # )

            # # Evaluate
            # vgg_test_loss, vgg_test_acc = vgg_model.evaluate(vg_test_gen)
            # print(
            #     f"VGG16 Test Accuracy: {vgg_test_acc * 100:.2f}% | Test Loss: {vgg_test_loss}"
            # )

            # save_to_result_json(
            #     {
            #         "arch": "VGG",
            #         "learn_rate": learning_rate,
            #         "batch_size": batch_size,
            #         "test_acc": vgg_test_acc,
            #         "test_loss": vgg_test_loss,
            #     }
            # )
            # =====================================================================

    return results


def display_misclassified_images(X_test, y_test, y_pred, num_images=5):
    """Display up to n misclassified images."""

    misclassified_indices = np.where(y_test != y_pred)[0]
    if len(misclassified_indices) == 0:
        print("No misclassified images!")
        return

    # Display misclassified images
    num_images = min(num_images, len(misclassified_indices))
    plt.figure(figsize=(15, num_images * 3))

    for i in range(num_images):
        index = misclassified_indices[i]
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(
            X_test[index], cmap="gray"
        )  # Adjust cmap if your images are in color
        plt.title(f"True Label: {y_test[index]}, Predicted: {y_pred[index]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def analyse_model(model, X_test, y_test, test_gen, class_map):
    """Runs the model evaluation & analysis routine on a given model and test set."""

    loss, accuracy = model.evaluate(test_gen)

    # Generate predictions
    y_pred_proba = model.predict(test_gen)  # Probabilities for each class
    y_pred = np.argmax(y_pred_proba, axis=1)  # Predicted class labels

    print(y_pred_proba)
    print(y_pred)
    print(y_test)
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate F1 Score (macro)
    from sklearn.metrics import f1_score

    f1_macro = f1_score(y_test, y_pred, average="macro")

    # ROC and AUC calculation
    # Convert y_test to one-hot encoded format
    y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]
    print(y_test_one_hot)

    # Compute ROC AUC score for multi-class
    auc_sklearn = roc_auc_score(
        y_test_one_hot, y_pred_proba, average="macro", multi_class="ovr"
    )

    # Print results
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"F1 Score (Macro): {f1_macro}")
    print(f"AUC (sklearn): {auc_sklearn}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Confusion Matrix plot
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_map,
        yticklabels=class_map,
    )
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=8)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()

    # Plot misclassified images
    display_misclassified_images(X_test, y_test, y_pred)


def main():
    print(tf.config.list_physical_devices("GPU"))  # Ensure we have GPU visible to TF
    tf.random.set_seed(SEED)  # Make training/testing runs repeatable.

    X, y, class_map = load_lfw_dataset(LFW_DIR)

    # Train-Test-Validation Split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # ================= UNCOMMENT FOR GRID SEARCH HYPER PARAMETER TUNING =================
    # grid_search_results = gridsearch(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     X_test,
    #     y_test,
    # )
    # print(grid_search_results)

    # # ================= UNCOMMENT FOR FINAL MODEL TRAINING =================
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    train_gen, val_gen, test_gen = get_datagens(
        efficientnet_preprocess,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        BATCH_SIZE,
    )

    # Fine tune
    model, history = fine_tuned_model(
        build_efficientnetb0(learning_rate=LEARNING_RATE),
        train_gen,
        val_gen,
    )
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss}")
    model.save("efficientb0.keras")
    joblib.dump(history, "en_model-training-history.joblib")
    plot_history(history, "EfficientNet-B0")

    # # ================= UNCOMMENT FOR MODEL ANALYSIS OUTPUT =================
    # model = tf.keras.models.load_model("resnet.keras")
    _, _, test_gen = get_datagens(
        efficientnet_preprocess,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        16,
    )
    analyse_model(model, X_test, y_test, test_gen, class_map)


if __name__ == "__main__":
    main()
