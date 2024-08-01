# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def svm_classifier(data):
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     model = SVC()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
    
#     return accuracy_score(y_test, predictions)

# # Example usage:
# data = read_csv('110ds.csv')
# accuracy = svm_classifier(data)
# print("Accuracy:", accuracy)


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def preprocess_data(data):
#     # If there are categorical features, encode them
#     # For now, assuming data is already numerical
#     return data

# def svm_classifier(data):
#     X = data.iloc[:, :-1]  # Features
#     y = data.iloc[:, -1]   # Target
    
#     # Feature scaling
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     model = SVC()  # Support Vector Classification
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
    
#     return accuracy_score(y_test, predictions)

# # Example usage:
# data = read_csv('110ds.csv')
# accuracy = svm_classifier(data)
# print("Accuracy:", accuracy)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Assuming data is already numerical for this example
    return data

def svm_classifier(data):
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    
    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = SVC()  # Support Vector Classification
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    
    return accuracy, conf_matrix, class_report

# Example usage:
data = read_csv('110ds.csv')
accuracy, conf_matrix, class_report = svm_classifier(data)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
