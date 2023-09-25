import numpy as np
# Define a simple dataset with binary attributes
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]
# Define the attributes and the target variable
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target_attribute = 'PlayTennis'
# Function to calculate entropy of a dataset
def entropy(dataset):
    labels = [row[-1] for row in dataset]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
# Function to calculate information gain for an attribute
def information_gain(dataset, attribute):
    entropy_before_split = entropy(dataset)
    values = np.unique([row[attributes.index(attribute)] for row in dataset])
    weighted_entropy_after_split = 0
    for value in values:
        subset = [row for row in dataset if row[attributes.index(attribute)] == value]
        weight = len(subset) / len(dataset)
        weighted_entropy_after_split += weight * entropy(subset)
    information_gain = entropy_before_split - weighted_entropy_after_split
    return information_gain
# Function to select the best attribute for splitting
def select_best_attribute(dataset, attributes):
    information_gains = [(attribute, information_gain(dataset, attribute)) for attribute in attributes]
    best_attribute = max(information_gains, key=lambda x: x[1])[0]
    return best_attribute
# Function to build the decision tree
def build_decision_tree(dataset, attributes):
    labels = [row[-1] for row in dataset]
    # If all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels[0]
    # If there are no attributes left, return the majority label
    if len(attributes) == 0:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        return majority_label
    # Otherwise, select the best attribute for splitting
    best_attribute = select_best_attribute(dataset, attributes)
    tree = {best_attribute: {}}
    # Remove the best attribute from the list of attributes
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    # Recursively build the subtree for each value of the best attribute
    for value in np.unique([row[attributes.index(best_attribute)] for row in dataset]):
        subset = [row for row in dataset if row[attributes.index(best_attribute)] == value]
        subtree = build_decision_tree(subset, remaining_attributes)
        tree[best_attribute][value] = subtree
    return tree
# Function to classify a new sample using the decision tree
def classify(tree, sample):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        value = sample[attributes.index(attribute)]
        if value in tree[attribute]:
            return classify(tree[attribute][value], sample)
        else:
            return "Unknown"
    else:
        return tree
# Build the decision tree
decision_tree = build_decision_tree(data, attributes)
# Classify a new sample
new_sample = ['Sunny', 'Hot', 'High', 'Weak']
result = classify(decision_tree, new_sample)
print(f"Prediction for {new_sample}: {result}")