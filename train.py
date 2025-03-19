import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# ✅ Define Paths
DATASET_DIR = "dataset/train"
MODEL_PATH = "model/sign_model.h5"
LABELS_PATH = "model/labels.json"

# ✅ Load Dataset with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical'
)

# ✅ Use MobileNetV2 as Base Model for Transfer Learning
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Freeze base layers

# ✅ Add Custom Layers for Sign Recognition
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
output_layer = Dense(len(train_generator.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# ✅ Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Train Model
model.fit(train_generator, epochs=10)

# ✅ Save Model & Labels
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model training complete! Model saved at:", MODEL_PATH)
