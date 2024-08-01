# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def build_decision_tree(data):
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     tree = DecisionTreeClassifier(criterion='entropy')
#     tree.fit(X, y)
#     return tree

# def print_tree(tree, feature_names):
#     tree_rules = export_text(tree, feature_names=feature_names)
#     print(tree_rules)


# # # Create sample data and save to CSV
# data = {
#     "Feature1": [1, 4, 7, 10, 13, 16],
#     "Feature2": [2, 5, 8, 11, 14, 17],
#     "Feature3": [3, 6, 9, 12, 15, 18],
#     "Target": [0, 1, 0, 1, 0, 1]
# }
# # df = pd.DataFrame(data)
# # df.to_csv("3ds.csv", index=False)

# # Example usage:
# # data = pd.read_csv('3ds.csv')
# tree = build_decision_tree(data)
# print_tree(tree, data.columns[:-1])




import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def build_decision_tree(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X, y)
    return tree

def print_tree(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)

# Read data from CSV
data = read_csv('3ds.csv')

# Build and print decision tree
tree = build_decision_tree(data)
print_tree(tree, list(data.columns[:-1]))



