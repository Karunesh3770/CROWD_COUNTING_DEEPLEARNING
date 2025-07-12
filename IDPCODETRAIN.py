import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import kagglehub

# 🔧 Enable memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 📌 CONSTANTS
BATCH_SIZE = 32
IMAGE_SIZE = (160, 160)
LEARNING_RATE = 0.001
EPOCHS = 20
CHECKPOINT_DIR = 'model_checkpoints'
MAX_SAMPLES = 200

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def create_density_map(label, image_size=IMAGE_SIZE):
    density_map = np.zeros(image_size, dtype=np.float32)
    count = float(label)
    if count > 0:
        center = (image_size[0] // 2, image_size[1] // 2)
        sigma = min(image_size) / 8
        x = np.arange(0, image_size[0], 1)
        y = np.arange(0, image_size[1], 1)
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
        gaussian = gaussian / np.sum(gaussian)
        density_map = gaussian * count
    return density_map

class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_path, labels_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES):
        self.batch_size = batch_size
        self.image_size = image_size
        self.images_data = np.load(image_path, mmap_mode='r')[:max_samples]
        self.labels_data = np.load(labels_path, mmap_mode='r')[:max_samples]
        self.total_images = self.images_data.shape[0]
        self._num_batches = (self.total_images + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.total_images)
        batch_images = np.zeros((batch_end - batch_start, *self.image_size, 3), dtype=np.float32)
        batch_density = np.zeros((batch_end - batch_start, *self.image_size, 1), dtype=np.float32)

        for i in range(batch_start, batch_end):
            img = self.images_data[i]
            img_resized = cv2.resize(img, self.image_size).astype(np.float32) / 255.0
            batch_images[i - batch_start] = img_resized
            density_map = create_density_map(self.labels_data[i], self.image_size)
            batch_density[i - batch_start] = np.expand_dims(density_map, axis=-1)

        return batch_images, batch_density

def create_crowd_counting_model(input_shape=(160, 160, 3)):
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outputs = tf.image.resize(outputs, input_shape[:2])
    return Model(inputs=inputs, outputs=outputs)

def train_model():
    # Download dataset using kagglehub
    base_path = kagglehub.model_download('fmena14/crowd-counting')
    image_path = os.path.join(base_path, 'images.npy')
    labels_path = os.path.join(base_path, 'labels.npy')

    # Prepare training data
    train_generator = MemoryEfficientDataGenerator(image_path, labels_path, max_samples=MAX_SAMPLES)
    print(f"Training on {train_generator.total_images} images")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch: {len(train_generator)}")

    # Create and compile model
    model = create_crowd_counting_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    # Define checkpoint callback
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_epoch_{epoch:02d}_loss_{loss:.4f}.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        verbose=1
    )

    # Define TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Training
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            checkpoint_callback,
            tensorboard_callback
        ]
    )

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"✅ Final model saved to {final_model_path}")

    return model, history

if __name__ == "__main__":
    print("🚀 Starting training process...")
    model, history = train_model()
    print("✅ Training completed!")