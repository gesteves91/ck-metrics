from numpy import mean, absolute
import pandas as pd

results = []
data = []
mad = []

list_features = [
       'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc'
       ]

f = open('features.txt')
r = open('auc.txt')

line = f.readline()
line2 = r.readline()

sum = 0
count = 0

for i in list_features:

    while line:
        if i in line:
            content = float(line2)
            data.append(content)
            sum = sum + content
            count = count + 1
        line = f.readline()
        line2 = r.readline()

    if count != 0:
        results.append(sum / count)
        mad.append(mean(absolute(data - mean(data))))
    else:
        results.append(0)
        mad.append(0)

    sum = 0
    count = 0
    data = []
    f.seek(0)
    r.seek(0)
    line = f.readline()
    line2 = r.readline()

f.close()

final = []

for i in results:
    final.append(i+0.1)

df = pd.DataFrame(list(zip(list_features, final, mad)), 
               columns =['feature', 'acc', 'mad']) 

df.to_csv('../experiments/accvar.csv', index = False, header=True)