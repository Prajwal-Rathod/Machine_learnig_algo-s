# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def knn_classifier(data, k):
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
    
#     return accuracy_score(y_test, predictions), predictions

# # Example usage:
# data = read_csv('8ds.csv')
# accuracy, predictions = knn_classifier(data, 3)
# print("Accuracy:", accuracy)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def knn_classifier(data, k):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Data normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    return accuracy, conf_matrix, class_report
data = read_csv('8ds.csv')
accuracy, conf_matrix, class_report = knn_classifier(data, 3)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
