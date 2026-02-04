from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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
    
    return model, score, matrix