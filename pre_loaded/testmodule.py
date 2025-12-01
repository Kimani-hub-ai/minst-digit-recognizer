import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the model
MODEL_PATH = "digit_model.keras"
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load MNIST test data
print("Loading MNIST test data...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess test data
x_test_processed = np.expand_dims(x_test, -1).astype('float32') / 255.0

# Evaluate on entire test set
print("\nEvaluating model on 10,000 test images...")
test_loss, test_accuracy = model.evaluate(x_test_processed, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")

# Predict on test set
print("\nGenerating predictions...")
predictions = model.predict(x_test_processed, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy per digit
print("\nAccuracy per digit:")
print("-" * 40)
for digit in range(10):
    mask = y_test == digit
    digit_accuracy = np.mean(predicted_labels[mask] == y_test[mask]) * 100
    count = np.sum(mask)
    print(f"Digit {digit}: {digit_accuracy:.2f}% ({count} samples)")

# Find misclassified examples
misclassified_indices = np.where(predicted_labels != y_test)[0]
print(f"\nTotal misclassified: {len(misclassified_indices)} out of {len(y_test)}")

# Show some misclassified examples
if len(misclassified_indices) > 0:
    print("\nShowing first 10 misclassified examples...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_indices) and i < 10:
            idx = misclassified_indices[i]
            ax.imshow(x_test[idx], cmap='gray')
            pred_label = predicted_labels[idx]
            true_label = y_test[idx]
            confidence = predictions[idx][pred_label] * 100
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                        fontsize=10, color='red')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved misclassified examples to 'misclassified_examples.png'")

# Show some correctly classified examples with high confidence
correct_indices = np.where(predicted_labels == y_test)[0]
confidences = np.max(predictions[correct_indices], axis=1)
high_conf_indices = correct_indices[np.argsort(confidences)[-10:]]

print("\nShowing 10 correctly classified examples (highest confidence)...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Correctly Classified Examples (High Confidence)', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(high_conf_indices):
        idx = high_conf_indices[i]
        ax.imshow(x_test[idx], cmap='gray')
        pred_label = predicted_labels[idx]
        confidence = predictions[idx][pred_label] * 100
        ax.set_title(f'Digit: {pred_label}\nConf: {confidence:.1f}%', 
                    fontsize=10, color='green')
        ax.axis('off')

plt.tight_layout()
plt.savefig('correct_examples.png', dpi=150, bbox_inches='tight')
print("✓ Saved correct examples to 'correct_examples.png'")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved confusion matrix to 'confusion_matrix.png'")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)