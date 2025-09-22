import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Simple retraining pipeline (transfer learning)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "training_dataset"  # put your dataset here

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Load pretrained MobileNetV2 for transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs("model", exist_ok=True)
with open("model/model.tflite", "wb") as f:
    f.write(tflite_model)

# Save labels
with open("labels.txt", "w") as f:
    for label in train_generator.class_indices.keys():
        f.write(label + "\n")

print("✅ Model retrained and saved to model/model.tflite")
