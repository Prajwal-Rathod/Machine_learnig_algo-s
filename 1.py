import csv

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def find_s(data):
    # Initialize the hypothesis to the first positive instance
    hypothesis = data[0][:-1]
    
    for instance in data:
        if instance[-1] == 'yes':  # Considering 'yes' as the positive class
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'
                    
    return hypothesis

# Example usage:
data = read_csv('1ds.csv')
hypothesis = find_s(data)
print("Final Hypothesis:", hypothesis)
