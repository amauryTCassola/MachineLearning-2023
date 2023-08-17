import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class ConfusionMatrix:
    """
    Implementação da matriz de confusão e suas métricas
    """
    def __init__(self, y_true: np.array, y_pred: np.array, labels: np.array):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.confusion_matrix = self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        confusion_matrix = np.zeros((len(self.labels), len(self.labels)))
        num_instances = len(self.y_pred)
        for idx in range(num_instances):
            confusion_matrix[self.y_true[idx]][self.y_pred[idx]] += 1
        return confusion_matrix

    def get_accuracy(self):
        accuracy = 0
        for idx in range(len(self.labels)):
            accuracy += self.confusion_matrix[idx][idx]
        return accuracy / np.sum(self.confusion_matrix)

    def get_precision(self):
        precision = 0
        # Precisão -> colunas
        for idx in range(len(self.labels)):
            # Evitar divisão por zero
            if np.sum(self.confusion_matrix[:, idx]) != 0:
                precision += self.confusion_matrix[idx][idx] / np.sum(self.confusion_matrix[:, idx])
        return precision / len(self.labels)

    def get_recall(self):
        recall = 0
        # Recall -> linhas
        for idx in range(len(self.labels)):
            # Evitar divisão por zero
            if np.sum(self.confusion_matrix[idx, :]) != 0:
                recall += self.confusion_matrix[idx][idx] / np.sum(self.confusion_matrix[idx, :])
        return recall / len(self.labels)

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * (precision * recall) / (precision + recall)


    def __str__(self):
        return f"Confusion Matrix: \n{self.confusion_matrix}\n" \
               f"Accuracy: {self.get_accuracy()*100:.2f}%\n" \
               f"Precision: {self.get_precision()*100:.2f}%\n" \
               f"Recall: {self.get_recall()*100:.2f}%\n" \
               f"F1 Score: {self.get_f1_score()*100:.2f}%\n"
        

if __name__ == "__main__":
    # Mesmo exemplo utilizando em https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    confusion_matrix = ConfusionMatrix(y_true, y_pred, labels=[0, 1, 2])
    print(confusion_matrix)
    confusion_matrix.show()

    print("De acordo com as funções do sklearn.metrics:")
    print(f"Sklearn Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Sklearn Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    print(f"Sklearn Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    print(f"Sklearn F1 Score: {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    