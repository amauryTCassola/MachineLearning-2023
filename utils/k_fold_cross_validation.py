import numpy as np
from functools import reduce
from sklearn.naive_bayes import GaussianNB
from confusion_matrix import ConfusionMatrix

class KFoldCrossValidation:
    """
    Implementação da k-fold cross-validation
    """
    def __init__(self, X: np.array, y: np.array, k: int, model):
        self.X = X
        self.y = y
        self.k = k
        self.model = model
        self.confusion_matrices = []
        self.avgAccuracy = 0
        self.stdAccuracy = 0
        self.avgPrecision = 0
        self.stdPrecision = 0
        self.avgRecall = 0
        self.stdRecall = 0
        self.avgF1 = 0
        self.stdF1 = 0
        self.compute_cross_validation()

    def merge_lists(self, list_of_lists):
        data = reduce(lambda a, b: np.concatenate((a, b)), list_of_lists)
        return data

    def divide_into_folds(self):
        where0 = np.where(self.y == 0)[0]
        where1 = np.where(self.y == 1)[0]
        
        folds0 = np.array_split(where0, k)
        folds1 = np.array_split(where1, k)

        #just to initialize the arrays
        X_folds = np.array_split(self.X, k)
        y_folds = np.array_split(self.y, k)
        
        for i in range(k):
            indexesList = merge_lists([folds0[i], folds1[i]])
            y_folds[i] = np.array(self.y[indexesList])
            X_folds[i] = np.array(self.X[indexesList])

        return X_folds, y_folds

    def compute_cross_validation(self):
        X_folds, y_folds = self.divide_into_folds()
        
        accuracies = np.zeros(k)
        precisions = np.zeros(k)
        recalls = np.zeros(k)
        f1_scores = np.zeros(k)

        for i in range(k):
            X_test = X_folds[i]
            y_test = y_folds[i]

            X_temp = X_folds.copy()
            X_temp.pop(i)
            X_train = self.merge_lists(X_temp)

            y_temp = y_folds.copy()
            y_temp.pop(i)
            y_train = self.merge_lists(y_temp)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            confusion_matrix = ConfusionMatrix(y_test, y_pred, labels=[0, 1])
            
            self.confusion_matrices.append(confusion_matrix)
            accuracies[i] = confusion_matrix.get_accuracy()
            precisions[i] = confusion_matrix.get_precision()
            recalls[i] = confusion_matrix.get_recall()
            f1_scores[i] = confusion_matrix.get_f1_score()

        self.avgAccuracy = np.average(accuracies)
        self.stdAccuracy = np.std(accuracies)
        self.avgPrecision = np.average(precisions)
        self.stdPrecision = np.std(precisions)
        self.avgRecall = np.average(recalls)
        self.stdRecall = np.std(recalls)
        self.avgF1 = np.average(f1_scores)
        self.stdF1 = np.std(f1_scores)


def merge_lists(list_of_lists):
    data = reduce(lambda a, b: np.concatenate((a, b)), list_of_lists)
    return data

if __name__ == "__main__":
    X = np.array([  [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1], ])

    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    k = 10

    gnb = GaussianNB()

    crossValidation = KFoldCrossValidation(X,y,k,gnb)

    print("Avg | std: ")
    print(f"Accuracy: {crossValidation.avgAccuracy*100:.2f}% | {crossValidation.stdAccuracy:.3f}")
    print(f"Precision: {crossValidation.avgPrecision*100:.2f}% | {crossValidation.stdPrecision:.3f}")
    print(f"Recall: {crossValidation.avgRecall*100:.2f}% | {crossValidation.stdRecall:.3f}")
    print(f"F1 Score: {crossValidation.avgF1*100:.2f}% | {crossValidation.stdF1:.3f}")

    
    for i in range(k):
        print("\n")
        print(f"Do modelo: {i}")
        print(f"Accuracy: {crossValidation.confusion_matrices[i].get_accuracy()*100:.2f}%")
        print(f"Precision: {crossValidation.confusion_matrices[i].get_precision()*100:.2f}%")
        print(f"Recall: {crossValidation.confusion_matrices[i].get_recall()*100:.2f}%")
        print(f"F1 Score: {crossValidation.confusion_matrices[i].get_f1_score()*100:.2f}%")


    