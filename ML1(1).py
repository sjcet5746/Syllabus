#1.Implement and demonstrate the FIND-S algorithm for finding the most specific
  #hypothesis based on a given set of training data samples. Read the training data from a CSV file
import csv
def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = dataset[i]
    return dataset
attributes = ['Sky','Temp','Humidity','Wind','Water','Forecast']
print('Attributes =',attributes)
num_attributes = len(attributes)
filename = "D:\Gskd\CSV\play_tennis.csv"
dataset = loadCsv(filename)
print(dataset)
hypothesis=['0'] * num_attributes
print("Intial Hypothesis")
print(hypothesis)
print("The Hypothesis are")
for i in range(len(dataset)):
    target = dataset[i][-1]
    if(target == 'Yes'):
        for j in range(num_attributes):
            if(hypothesis[j]=='0'):
                hypothesis[j] = dataset[i][j]
            if(hypothesis[j]!= dataset[i][j]):
                hypothesis[j]='?'
        print(i+1,'=',hypothesis)
print("Final Hypothesis")
print(hypothesis)