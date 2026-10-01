
import matplotlib.pyplot as plt

# Model names
models = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Decision Tree",
    "KNN",
    "XGBoost",
    "AdaBoost",
    "Tuned Decision Tree"
]

# Accuracy scores
accuracy = [
    70.78,
    77.92,
    74.03,
    68.18,
    75.32,
    76.00,
    76.00,
    77.27
]

# F1-scores
f1_score = [
    54.50,
    66.00,
    60.00,
    51.50,
    63.50,
    64.10,
    63.40,
    70.09
]

# ROC-AUC scores
roc_auc = [
    81.30,
    81.79,
    79.64,
    63.57,
    78.86,
    82.31,
    None,
    79.92
]

# Create the chart
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.plot(
    models,
    accuracy,
    marker="o",
    label="Accuracy"
)

# Plot F1-score
plt.plot(
    models,
    f1_score,
    marker="o",
    label="F1-score"
)

# Remove the missing ROC-AUC value
roc_models = [
    model
    for model, score in zip(models, roc_auc)
    if score is not None
]

roc_scores = [
    score
    for score in roc_auc
    if score is not None
]

# Plot ROC-AUC
plt.plot(
    roc_models,
    roc_scores,
    marker="o",
    label="ROC-AUC"
)

# Add labels and title
plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score (%)")

# Rotate model names so they are easier to read
plt.xticks(rotation=45, ha="right")

# Show legend
plt.legend()

# Add grid
plt.grid(True)

# Prevent labels from being cut off
plt.tight_layout()

# Save the chart inside the images folder
plt.savefig(
    "images/model_comparison.png",
    dpi=300
)

# Display the chart
plt.show()




# ============================================================
# Confusion Matrix
# ============================================================

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Actual confusion matrix values from the final model
cm = np.array([
    [83, 17],
    [15, 39]
])

# Create the confusion matrix visualization
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot()

# Add title
plt.title("Final Model Confusion Matrix")

# Adjust layout
plt.tight_layout()

# Save the image
plt.savefig(
    "images/confusion_matrix.png",
    dpi=300
)

# Display the image
plt.show()