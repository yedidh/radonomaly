import numpy as np
from scipy.io import arff
from sklearn import preprocessing
from sphered_cumulative_radon_features import anomaly_score

def process(path, is_sad=False):
    data, meta = arff.loadarff(open(path,'r'))
    N = len(data)
    D = len(data[0][0])
    M = len(data[0][0][0])
    dataset = np.zeros((N,D,M),dtype='float')
    for n in range(N):
        for d in range(D):
            for m in range(M):
                dataset[n, d, m] = float(data[n][0][d][m])
    targets = [data[i][1] for i in range(N)]
    if is_sad:
        l = np.argmax(np.isnan(dataset[:, 0, :]), axis=1)
        idx = np.where(np.logical_and(l <=50, l >= 20))[0]
        dataset = dataset[idx]
        targets = [targets[i] for i in idx]
    dataset[np.isnan(dataset)] = 0
    return dataset.astype('float32'), targets

ds = 'ct'
is_sad = False
if ds == 'epsy':
    train_path = 'Epilepsy/Epilepsy_TRAIN.arff'
    test_path = 'Epilepsy/Epilepsy_TEST.arff'
if ds== 'ct':
    train_path = 'CharacterTrajectories/CharacterTrajectories_TRAIN.arff'
    test_path = 'CharacterTrajectories/CharacterTrajectories_TEST.arff'
if ds == 'na':
    train_path = 'NATOPS/NATOPS_TRAIN.arff'
    test_path = 'NATOPS/NATOPS_TEST.arff'
if ds == 'rs':
    train_path = 'RacketSports/RacketSports_TRAIN.arff'
    test_path = 'RacketSports/RacketSports_TEST.arff'
if ds == 'sad':
    train_path = 'SpokenArabicDigits/SpokenArabicDigits_TRAIN.arff'
    test_path = 'SpokenArabicDigits/SpokenArabicDigits_TEST.arff'
    is_sad = True

train_dataset, train_targets = process(train_path, is_sad)
test_dataset, test_targets = process(test_path, is_sad)

le = preprocessing.LabelEncoder()
le.fit(train_targets)
train_targets_num = le.transform(train_targets)
test_targets_num = le.transform(test_targets)

auc_sum = 0
n_classes = np.max(test_targets_num)+1
for c in range(n_classes):
    auc = anomaly_score(train_dataset[train_targets_num == c], test_dataset, test_targets_num != c)
    auc_sum += auc
    print("Class: ", c, " ROCAUC: ", auc)
print("Average ROCAUC: ", auc_sum / n_classes)
