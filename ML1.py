#1. Implement and demonstrate the FIND-S algorithm for finding the most specific
    #hypothesis based on a given set of training data samples. Read the training data from a CSV file
import random
import csv
def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        traindata = []
        for row in datareader:
            traindata.append(row)
    return (traindata)

h=['phi','phi','phi','phi','phi','phi']
data=read_data("D:\Gskd\CSV\play_tennis.csv")
def isConsistent(h,d):
    if len(h)!=len(d)-1:
        print('Number of attributes are not same in hypothesis.')
        return False
    else:
        matched=0
        for i in range(len(h)):
            if ( (h[i]==d[i]) | (h[i]=='any') ):
                matched=matched+1
            if matched==len(h):
                return True
            else:
                return False
def makeConsistent(h,d):
    for i in range(len(h)):
        if((h[i] == 'phi')):
            h[i]=d[i]
        elif(h[i]!=d[i]):
            h[i]='any'
    return h
print('Begin : Hypothesis :',h)
print('==========================================')
for d in data:
    if d[len(d)-1]=='Yes':
        if ( isConsistent(h,d)):
            pass
        else:
            h=makeConsistent(h,d)
    print ('Training data :',d)
    print ('Updated Hypothesis :',h)
    print()
    print(' ')
print('==========================================')
print('maximally sepcific data set End: Hypothesis :',h)