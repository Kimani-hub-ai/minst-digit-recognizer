import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

print("Loading MNIST dataset...")
# -----------------------------
# Load MNIST dataset
# -----------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")

# -----------------------------
# Preprocessing with augmentation
# -----------------------------
def preprocess(x, y):
    """Simple preprocessing - normalize to [0,1]"""
    x = tf.expand_dims(x, -1)  # Add channel dimension
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

def preprocess_augmented(x, y):
    """Preprocessing with data augmentation"""
    x = tf.expand_dims(x, -1)
    x = tf.cast(x, tf.float32) / 255.0
    
    # Random shifts (±2 pixels)
    x = tf.image.random_crop(
        tf.pad(x, [[2, 2], [2, 2], [0, 0]], mode='CONSTANT'),
        [28, 28, 1]
    )
    
    # Random brightness
    x = tf.image.random_brightness(x, max_delta=0.1)
    
    return x, y

# -----------------------------
# Create datasets
# -----------------------------
BATCH_SIZE = 128

print("\nPreparing training dataset with augmentation...")
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess_augmented, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Preparing validation dataset...")
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Build Custom CNN for MNIST
# -----------------------------
print("\nBuilding CNN model architecture...")

model = models.Sequential([
    # Input layer
    layers.Input(shape=(28, 28, 1)),
    
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Output layer
    layers.Dense(10, activation='softmax')
], name='MNIST_CNN')

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*70)
model.summary()
print("="*70 + "\n")

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -----------------------------
# Train the model
# -----------------------------
print("Starting training...")
print("="*70)

EPOCHS = 25

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Save final model
# -----------------------------
MODEL_SAVE_PATH = "digit_model.keras"
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

# -----------------------------
# Evaluate model
# -----------------------------
print("\nEvaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f"✓ Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"✓ Validation Loss: {val_loss:.4f}")

# -----------------------------
# Plot training history
# -----------------------------
print("\nGenerating training plots...")

plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training', linewidth=2, color='#2E86AB')
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#A23B72')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0.9, 1.0])

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training', linewidth=2, color='#2E86AB')
plt.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#A23B72')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_history.png', dpi=150, bbox_inches='tight')
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')  # Also save in root
print("✓ Training plots saved to 'training_history.png'")

# -----------------------------
# Quick test on some samples
# -----------------------------
print("\nTesting on random validation samples...")
test_indices = np.random.choice(len(x_val), 10, replace=False)
test_samples = x_val[test_indices]
test_labels = y_val[test_indices]

# Preprocess
test_samples_processed = np.expand_dims(test_samples, -1).astype('float32') / 255.0

# Predict
predictions = model.predict(test_samples_processed, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample predictions:")
print("-" * 50)
correct = 0
for i in range(10):
    confidence = predictions[i][predicted_labels[i]] * 100
    is_correct = predicted_labels[i] == test_labels[i]
    if is_correct:
        correct += 1
    symbol = "✓" if is_correct else "✗"
    print(f"{symbol} True: {test_labels[i]}, Predicted: {predicted_labels[i]} (Confidence: {confidence:.1f}%)")

print("-" * 50)
print(f"Quick test accuracy: {correct}/10 ({correct*10}%)")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print(f"Expected accuracy on hand-drawn digits: 95-99%")
print("="*70)