import time
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

'''network_path = 'MSCMF/'
fpr_o = np.loadtxt(network_path + 'fpr.csv', delimiter=',')
tpr_o = np.loadtxt(network_path + 'tpr.csv', delimiter=',')
recall_o = np.loadtxt(network_path + 'recall.csv', delimiter=',')
precision_o = np.loadtxt(network_path + 'precision.csv', delimiter=',')'''

network_path = ''
fpr = np.loadtxt(network_path +'fpr.csv', delimiter=',')
tpr = np.loadtxt(network_path +'tpr.csv', delimiter=',')
recall = np.loadtxt(network_path +'recall.csv', delimiter=',')
precision = np.loadtxt(network_path +'precision.csv', delimiter=',')

'''roc, prc = [], []
auc, aupr = [], []
color = ['blue', 'orange', 'gray', 'gold', 'royalblue']
label = ['MSCMF', 'TL_HGBI', 'DTINet', 'NeoDTI', 'GADTI']

for i in range(5):
    roc.append(np.loadtxt('roc' + str(i + 1) + '.csv'))
    prc.append(np.loadtxt('prc' + str(i + 1) + '.csv'))

for i in range(5):
    for x, y in roc[i]:
        test_auc = 0.
        prev_x = 0.
        if x != prev_x:
            test_auc += (x - prev_x) * y
            prev_x = x
        auc.append(test_auc)

    for x, y in prc[i]:
        test_aupr = 0.
        prev_x = 0.
        if x != prev_x:
            test_aupr += (x - prev_x) * y
            prev_x = x
        aupr.append(test_aupr)

plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
for i in range(5):
    x = [x_[0] for x_ in roc[i]]
    y = [x_[1] for x_ in roc[i]]
    plt.plot(x, y, linewidth='1', label=label[i], color=color[i])
    # plt.plot(x, y, linewidth='1', label="test", color=' coral ', linestyle=':', marker='|')
plt.legend()
plt.show()

plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
for i in range(5):
    x = [x_[0] for x_ in prc[i]]
    y = [x_[1] for x_ in prc[i]]
    plt.plot(x, y, linewidth='1', label=label[i], color=color[i])
plt.legend()
plt.show()'''

'''fpr = []
tpr = []
recall=[]
precision=[]

for i in (range(len(fpr_o))):
    if i % 10== 0:
        fpr.append(fpr_o[i])
        tpr.append(tpr_o[i])

for i in (range(len(recall_o))):
    if i % 10== 0:
        recall.append(recall_o[i])
        precision.append(precision_o[i])

np.savetxt('fpr1.csv', fpr, fmt='%-.4f', delimiter=',')
np.savetxt('tpr1.csv', tpr, fmt='%-.4f', delimiter=',')
np.savetxt('recall1.csv', recall, fmt='%-.4f', delimiter=',')
np.savetxt('precision1.csv', precision, fmt='%-.4f', delimiter=',')'''

auc = 0.
prev_x = 0.
for i in range(len(fpr)):
    if fpr[i] != prev_x:
        auc += (fpr[i] - prev_x) * tpr[i]
        prev_x = fpr[i]

aupr = 0.
prev_x = 0.
for i in range(len(recall)):
    if recall[i] != prev_x:
        aupr += (recall[i] - prev_x) * precision[i]
        prev_x = recall[i]

plt.title("ROC curve of %s (AUC = %.4f)" % ('GADTI', auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(fpr, tpr, 'r')
plt.show()

plt.title("PR curve of %s (AUPR = %.4f)" % ('GADTI', aupr))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision)
plt.show()
