import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Load preprocessed data
data = np.load("preprocessed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Load the trained model
model = tf.keras.models.load_model("saved_models/best_model.h5")

# Evaluate the model on the test set
print("Evaluate: Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)  # If y_test is one-hot encoded

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Generate classification report (precision, recall, f1-score)
report = classification_report(
    y_test_classes, y_pred_classes, target_names=["Defective", "Good"])
print("Classification Report:")
print(report)

# Calculate Precision, Recall, and F1-score
precision = precision_score(y_test_classes, y_pred_classes, average='binary')
recall = recall_score(y_test_classes, y_pred_classes, average='binary')
f1 = f1_score(y_test_classes, y_pred_classes, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Calculate Intersection over Union (IoU) for both classes


def calculate_iou(y_true, y_pred, class_idx):
    intersection = np.sum(np.logical_and(
        y_true == class_idx, y_pred == class_idx))
    union = np.sum(np.logical_or(y_true == class_idx, y_pred == class_idx))
    return intersection / union if union != 0 else 0


iou_defective = calculate_iou(y_test_classes, y_pred_classes, 1)
iou_good = calculate_iou(y_test_classes, y_pred_classes, 0)
print(f"IoU (Defective): {iou_defective:.4f}")
print(f"IoU (Good): {iou_good:.4f}")

# Visualizing the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Defective", "Good"], yticklabels=["Defective", "Good"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Save the confusion matrix plot
cm_plot_path = "results/confusion_matrix.png"
# Create the results directory if it doesn't exist
os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
plt.savefig(cm_plot_path)
print(f"Confusion matrix saved as {cm_plot_path}")

# Visualizing Precision, Recall, and F1-score
metrics = [precision, recall, f1]
metric_names = ['Precision', 'Recall', 'F1-Score']
plt.figure(figsize=(8, 6))
plt.bar(metric_names, metrics, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Precision, Recall, and F1-Score')
plt.ylabel('Score')
plt.show()

# Save the metrics plot
metrics_plot_path = "results/metrics_plot.png"
plt.savefig(metrics_plot_path)
print(f"Metrics plot saved as {metrics_plot_path}")
