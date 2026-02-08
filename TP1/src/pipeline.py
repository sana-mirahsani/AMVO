from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def split_train_test(X, y, test_size=0.20, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def fit_predict(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred

def eval_accuracy(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)
    return score

def eval_cm(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    return matrix

def pipeline_histogram_joint(X,y, model, test_size, random_state):

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    
    model, y_pred = fit_predict(model, X_train, X_test, y_train)
    
    score = eval_accuracy(y_test, y_pred)
    
    matrix = eval_cm(y_test, y_pred)
    
    # plot matrix confusion
    plot_confusion_matrix(cm= matrix, class_names=list(set(y)), normalized=False, title="Confusion Matrix")
    return model, score, matrix

def plot_confusion_matrix(cm, class_names, normalized=False, title="Confusion Matrix"):
    """
    Plots a confusion matrix.
    
    Parameters:
    - cm: numpy array of shape (n_classes, n_classes)
          Confusion matrix
    - class_names: list or array of class labels
    - normalized: bool, whether the confusion matrix is normalized
    - title: string, title of the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalized else "d"
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()