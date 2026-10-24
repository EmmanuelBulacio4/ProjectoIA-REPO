from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomRotation,
    RandomTranslation,
)
from tensorflow.keras.models import Sequential


DATASET_DIR = Path(
    r"C:\Users\Emman\Desktop\Proyecto IA\imagenes_redimensionadas\dataset"
)
IMG_SIZE = (480, 640)
BATCH_SIZE = 16
EPOCHS = 5
CLASS_NAMES = ["0", "1"]  # 0 = fine, 1 = hook
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

MODEL_PATH = Path("fine_hook_model.keras")
TFLITE_PATH = Path("fine_hook_model.tflite")


def load_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=123,
    )
    return dataset.map(
        lambda images, labels: (preprocess_input(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

def get_class_weights() -> dict[int, float]:
    counts = {
        class_index: sum(
            1 for path in (DATASET_DIR / "train" / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_index, class_name in enumerate(CLASS_NAMES)
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError("Cada carpeta de clase en train debe contener imágenes.")

    total = sum(counts.values())
    return {
        class_index: total / (len(CLASS_NAMES) * count)
        for class_index, count in counts.items()
    }


def build_model() -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    return Sequential(
        [
            tf.keras.Input(shape=(*IMG_SIZE, 3)),
            RandomRotation(factor=0.25, fill_mode="reflect"),
            RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="reflect",
            ),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )


def main() -> None:
    train_ds = load_dataset("train", shuffle=True)
    val_ds = load_dataset("val", shuffle=False)
    test_ds = load_dataset("test", shuffle=False)

    print("Clases:", CLASS_NAMES)
    print("Pesos por clase:", get_class_weights())

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.2,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=get_class_weights(),
        callbacks=callbacks,
    )

    print("\nEvaluación final en test:")
    model.evaluate(test_ds, return_dict=True)

    true_labels = tf.concat([labels for _, labels in test_ds], axis=0)
    probabilities = model.predict(test_ds)
    predicted_labels = tf.argmax(probabilities, axis=1)
    print("Matriz de confusión (filas = real, columnas = predicha):")
    print(
        tf.math.confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes=len(CLASS_NAMES),
        ).numpy()
    )

    model.save(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    TFLITE_PATH.write_bytes(converter.convert())
    print(f"\nModelo TensorFlow guardado en: {MODEL_PATH.resolve()}")
    print(f"Modelo TensorFlow Lite guardado en: {TFLITE_PATH.resolve()}")


if __name__ == "__main__":
    main()
