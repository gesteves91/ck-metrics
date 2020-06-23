import pandas as pd

features = open('features.txt')
accuracies = open('auc.txt')

line = features.readline()
line2 = accuracies.readline()

feat = []
acc = []

while line:
    feat.append(line)
    acc.append(line2)
    line = features.readline()
    line2 = accuracies.readline()

df = pd.DataFrame(list(zip(feat, acc)), columns =['feature', 'acc'])

df.to_csv('../experiments/models.csv', index = False, header=True)