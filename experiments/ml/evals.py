from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
logger = getLogger(__name__)

def evaluate(perdicted_class, predicted_proba, true_class, writer: SummaryWriter, epoch: int):
    accuracy = accuracy_score(true_class, perdicted_class)
    precision = precision_score(true_class, perdicted_class)
    recall = recall_score(true_class, perdicted_class)
    f1 = f1_score(true_class, perdicted_class)
    roc_auc = roc_auc_score(true_class, predicted_proba)

    writer.add_scalar('Eval/Accuracy', accuracy, epoch)
    writer.add_scalar('Eval/Precision', precision, epoch)
    writer.add_scalar('Eval/Recall', recall, epoch)
    writer.add_scalar('Eval/F1_Score', f1, epoch)
    writer.add_scalar('Eval/ROC_AUC', roc_auc, epoch)

    logger.info(f"Eval Metrics - Epoch {epoch}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
