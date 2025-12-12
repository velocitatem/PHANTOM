from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

logger = getLogger(__name__)

def evaluate(perdicted_class, predicted_proba, true_class, writer: SummaryWriter, epoch: int):
    accuracy = accuracy_score(true_class, perdicted_class)
    precision = precision_score(true_class, perdicted_class, zero_division=0)
    recall = recall_score(true_class, perdicted_class, zero_division=0)
    f1 = f1_score(true_class, perdicted_class, zero_division=0)
    roc_auc = roc_auc_score(true_class, predicted_proba)

    writer.add_scalar('Eval/Accuracy', accuracy, epoch)
    writer.add_scalar('Eval/Precision', precision, epoch)
    writer.add_scalar('Eval/Recall', recall, epoch)
    writer.add_scalar('Eval/F1_Score', f1, epoch)
    writer.add_scalar('Eval/ROC_AUC', roc_auc, epoch)

    # confusion matrix
    cm = confusion_matrix(true_class, perdicted_class)
    tn, fp, fn, tp = cm.ravel()
    writer.add_scalar('Eval/TrueNeg', tn, epoch)
    writer.add_scalar('Eval/FalsePos', fp, epoch)
    writer.add_scalar('Eval/FalseNeg', fn, epoch)
    writer.add_scalar('Eval/TruePos', tp, epoch)

    # specificity and sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # same as recall/TPR
    writer.add_scalar('Eval/Specificity', specificity, epoch)
    writer.add_scalar('Eval/Sensitivity', sensitivity, epoch)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.matshow(cm, cmap='Blues', alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'Confusion Matrix (Epoch {epoch})')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Human', 'Agent'])
    ax1.set_yticklabels(['Human', 'Agent'])

    # ROC curve
    fpr, tpr, _ = roc_curve(true_class, predicted_proba)
    ax2.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img_arr = np.array(img)
    writer.add_image('Eval/Metrics', img_arr, epoch, dataformats='HWC')
    plt.close()

    logger.info(f"Eval {epoch}: Acc={accuracy:.4f} Prec={precision:.4f} Rec={recall:.4f} F1={f1:.4f} AUC={roc_auc:.4f}")
