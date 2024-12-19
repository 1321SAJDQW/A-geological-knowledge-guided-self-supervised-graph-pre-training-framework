import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

file_path = 'Prediction_GKGP.csv'

df = pd.read_csv(file_path)

df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

y_true = df['label']
y_scores = df['Prediction_Probability']

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
