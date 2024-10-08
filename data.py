import scipy.io as scio
import numpy as np
from numpy import array
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

X_2dA = [[0 for i in range(8)] for j in range(29)]
X_2dB = [[0 for i in range(8)] for j in range(49)]
X_2dC = [[0 for i in range(8)] for j in range(51)]
X_2dD = [[0 for i in range(8)] for j in range(32)]
XXX_2d = [[0 for i in range(13858)] for j in range(161)]

Y_2d = [0 for i in range(161)]
X_2d = [[0 for i in range(8)] for j in range(161)]
df = pd.read_csv("./ADNI_MUL_T1_6_24_2024.csv", encoding="ANSI")  # Read data
df_array = np.array(df)
df_list = df_array.tolist()
dfA = pd.read_csv("./ADNI_MUL_add_T1_7_28_2024Z.csv", encoding="ANSI")
df_arrayA = np.array(dfA)
df_listA = df_arrayA.tolist()
dfAGE = []
dfAGE_array = np.zeros((6,161))
# dfACC = pd.read_csv("./C_loss.csv", encoding="UTF-8")
# dfACC_array = np.array(dfACC)
# dfAGE = pd.read_csv("./AGEmessIIM3V.CSV",encoding="UTF-8")
# dfAGE_array = np.array(dfAGE)
dfACC = []
dfACC_array = np.zeros((1,161))
dfAGE_list=[]
dfACC_list=[]


def data_init():
    data = {}
    global X_2dA
    global X_2dB
    global X_2dC
    global X_2dD
    global Y_2d
    global dfAGE
    global dfAGE_array
    global dfACC
    global dfACC_array
    Y_2dA = [0 for i in range(29)]
    Y_2dB = [0 for i in range(49)]
    Y_2dC = [0 for i in range(51)]
    Y_2dD = [0 for i in range(32)]

    tA, tB, tC, tD = 0, 0, 0, 0  # Read out the relevant data
    for i in range(1, 120):
        if df_list[i][4] == '1':
            X_2dA[tA][0] = np.array(df_list[i][20])
            X_2dA[tA][1] = np.array(df_list[i][22])
            X_2dA[tA][2] = np.array(df_list[i][24])
            X_2dA[tA][3] = np.array(df_list[i][26])
            X_2dA[tA][4] = np.array(df_list[i][28])
            X_2dA[tA][5] = np.array(df_list[i][29])
            X_2dA[tA][6] = np.array(df_list[i][30])
            X_2dA[tA][7] = np.array(df_list[i][19])
            Y_2dA[tA] = np.array(df_list[i][4])
            tA = tA + 1
        print(i, tA)
    print(tA)
    for i in range(1, 120):
        if df_list[i][4] == '2':
            X_2dB[tB][0] = np.array(df_list[i][20])
            X_2dB[tB][1] = np.array(df_list[i][22])
            X_2dB[tB][2] = np.array(df_list[i][24])
            X_2dB[tB][3] = np.array(df_list[i][26])
            X_2dB[tB][4] = np.array(df_list[i][28])
            X_2dB[tB][5] = np.array(df_list[i][29])
            X_2dB[tB][6] = np.array(df_list[i][30])
            X_2dB[tB][7] = np.array(df_list[i][19])
            Y_2dB[tB] = np.array(df_list[i][4])
            tB = tB + 1
        print(i, tB)
    print(tB)
    for i in range(1, 120):
        if df_list[i][4] == '3':
            X_2dC[tC][0] = np.array(df_list[i][20])
            X_2dC[tC][1] = np.array(df_list[i][22])
            X_2dC[tC][2] = np.array(df_list[i][24])
            X_2dC[tC][3] = np.array(df_list[i][26])
            X_2dC[tC][4] = np.array(df_list[i][28])
            X_2dC[tC][5] = np.array(df_list[i][29])
            X_2dC[tC][6] = np.array(df_list[i][30])
            X_2dC[tC][7] = np.array(df_list[i][19])
            Y_2dC[tC] = np.array(df_list[i][4])
            tC = tC + 1
        print(i, tC)
    print(tC)
    for i in range(1, 120):
        if df_list[i][4] == '4':
            X_2dD[tD][0] = np.array(df_list[i][20])
            X_2dD[tD][1] = np.array(df_list[i][22])
            X_2dD[tD][2] = np.array(df_list[i][24])
            X_2dD[tD][3] = np.array(df_list[i][26])
            X_2dD[tD][4] = np.array(df_list[i][28])
            X_2dD[tD][5] = np.array(df_list[i][29])
            X_2dD[tD][6] = np.array(df_list[i][30])
            X_2dD[tD][7] = np.array(df_list[i][19])
            Y_2dD[tD] = np.array(df_list[i][4])
            tD = tD + 1
        print(i, tD)
    print(tD)
    for i in range(1, 51):
        if df_listA[i][4] == '1':
            X_2dA[tA][0] = np.array(df_listA[i][20])
            X_2dA[tA][1] = np.array(df_listA[i][22])
            X_2dA[tA][2] = np.array(df_listA[i][24])
            X_2dA[tA][3] = np.array(df_listA[i][26])
            X_2dA[tA][4] = np.array(df_listA[i][28])
            X_2dA[tA][5] = np.array(df_listA[i][29])
            X_2dA[tA][6] = np.array(df_listA[i][30])
            X_2dA[tA][7] = np.array(df_listA[i][19])
            Y_2dA[tA] = np.array(df_listA[i][4])
            tA = tA + 1
        print(i, tA)
    print(tA)
    for i in range(1, 51):
        if df_listA[i][4] == '2':
            X_2dB[tB][0] = np.array(df_listA[i][20])
            X_2dB[tB][1] = np.array(df_listA[i][22])
            X_2dB[tB][2] = np.array(df_listA[i][24])
            X_2dB[tB][3] = np.array(df_listA[i][26])
            X_2dB[tB][4] = np.array(df_listA[i][28])
            X_2dB[tB][5] = np.array(df_listA[i][29])
            X_2dB[tB][6] = np.array(df_listA[i][30])
            X_2dB[tB][7] = np.array(df_listA[i][19])
            Y_2dB[tB] = np.array(df_listA[i][4])
            tB = tB + 1
        print(i, tB)
    print(tB)
    for i in range(1, 51):
        if df_listA[i][4] == '3':
            X_2dC[tC][0] = np.array(df_listA[i][20])
            X_2dC[tC][1] = np.array(df_listA[i][22])
            X_2dC[tC][2] = np.array(df_listA[i][24])
            X_2dC[tC][3] = np.array(df_listA[i][26])
            X_2dC[tC][4] = np.array(df_listA[i][28])
            X_2dC[tC][5] = np.array(df_listA[i][29])
            X_2dC[tC][6] = np.array(df_listA[i][30])
            X_2dC[tC][7] = np.array(df_listA[i][19])
            Y_2dC[tC] = np.array(df_listA[i][4])
            tC = tC + 1
        print(i, tC)
    print(tC)
    for i in range(1, 51):
        if df_listA[i][4] == '4':
            X_2dD[tD][0] = np.array(df_listA[i][20])
            X_2dD[tD][1] = np.array(df_listA[i][22])
            X_2dD[tD][2] = np.array(df_listA[i][24])
            X_2dD[tD][3] = np.array(df_listA[i][26])
            X_2dD[tD][4] = np.array(df_listA[i][28])
            X_2dD[tD][5] = np.array(df_listA[i][29])
            X_2dD[tD][6] = np.array(df_listA[i][30])
            X_2dD[tD][7] = np.array(df_listA[i][19])
            Y_2dD[tD] = np.array(df_listA[i][4])
            tD = tD + 1
        print(i, tD)
    print(tD)

    # Imputation
    print(nan_euclidean_distances(X_2dA, X_2dA))
    print(X_2dA)
    imputer = KNNImputer(n_neighbors=15)
    X_2dA = imputer.fit_transform(X_2dA)
    print(X_2dA)
    print('Y:')
    print(Y_2dA)
    print(nan_euclidean_distances(X_2dB, X_2dB))
    print(X_2dB)
    imputer = KNNImputer(n_neighbors=25)
    X_2dB = imputer.fit_transform(X_2dB)
    print(X_2dB)
    print(nan_euclidean_distances(X_2dC, X_2dC))
    print(X_2dC)
    imputer = KNNImputer(n_neighbors=26)
    X_2dC = imputer.fit_transform(X_2dC)
    print(X_2dC)
    # Distance calculation
    print(nan_euclidean_distances(X_2dD, X_2dD))
    print(X_2dD)
    imputer = KNNImputer(n_neighbors=18)
    X_2dD = imputer.fit_transform(X_2dD)
    print(X_2dD)
    print(len(X_2dA), len(X_2dB), len(X_2dC), len(X_2dD))
    I = 0
    # Import fMRI modal data
    for i in range(120):
        if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
            continue;
        if i < 10:
            data[I] = scio.loadmat(
                './resultsROI_Subject00' + str(i) + '_Condition001.mat')
        elif i >= 10 and i <= 99:
            data[I] = scio.loadmat(
                './resultsROI_Subject0' + str(i) + '_Condition001.mat')
        else:
            data[I] = scio.loadmat(
                './resultsROI_Subject' + str(i) + '_Condition001.mat')
        I = I + 1
        print(i, I)
    for i in range(51):
        if (i == 0 or i == 31 or i == 32):
            continue
        if i < 10:
            data[I] = scio.loadmat(
                './resultsROI_Subject00' + str(i) + '_Condition001.mat')
        elif i >= 10 and i <= 99:
            data[I] = scio.loadmat('./resultsROI_Subject0' + str(i) + '_Condition001.mat')
        else:
            data[I] = scio.loadmat('./resultsROI_Subject' + str(i) + '_Condition001.mat')
        I = I + 1
        print(i, I)

    X = {}
    for i in range(161):
        X[i] = data[i]['Z']
    t = 0
    for i in range(161):
        print(i)
        t = 0
        for k in range(0, 164):
            for l in range(k + 1, 164):
                XXX_2d[i][t] = np.array(X[i][k, l])
                t = t + 1
        for k in range(0, 164):
            for l in range(164, 167):
                XXX_2d[i][t] = np.array(X[i][k, l])
                t = t + 1
    print(t)
    print('XXX_2d:',XXX_2d[0])
    tAA, tBB, tCC, tDD = 0, 0, 0, 0
    global X_2d
    global Y_2d
    t = 0

    for i in range(1, 120):
        if np.array(df_list[i][4]) == '0':
            continue;
        if np.array(df_list[i][4]) == '1':
            X_2d[t] = np.array(X_2dA[tAA])
            Y_2d[t] = np.array(Y_2dA[tAA])
            print(Y_2d[t], Y_2dA[tAA])
            tAA = tAA + 1
            t = t + 1
        if np.array(df_list[i][4]) == '2':
            X_2d[t] = np.array(X_2dB[tBB])
            Y_2d[t] = np.array(Y_2dB[tBB])
            print(Y_2d[t], Y_2dB[tBB])
            tBB = tBB + 1
            t = t + 1
        if np.array(df_list[i][4]) == '3':
            X_2d[t] = np.array(X_2dC[tCC])
            Y_2d[t] = np.array(Y_2dC[tCC])
            print(Y_2d[t], Y_2dC[tCC])
            tCC = tCC + 1
            t = t + 1
        if np.array(df_list[i][4]) == '4':
            X_2d[t] = np.array(X_2dD[tDD])
            Y_2d[t] = np.array(Y_2dD[tDD])
            print(Y_2d[t], Y_2dD[tDD])
            tDD = tDD + 1
            t = t + 1
    for i in range(1, 51):
        if np.array(df_listA[i][4]) == '0':
            continue;
        if np.array(df_listA[i][4]) == '1':
            X_2d[t] = np.array(X_2dA[tAA])
            Y_2d[t] = np.array(Y_2dA[tAA])
            print(Y_2d[t], Y_2dA[tAA])
            tAA = tAA + 1
            t = t + 1
        if np.array(df_listA[i][4]) == '2':
            X_2d[t] = np.array(X_2dB[tBB])
            Y_2d[t] = np.array(Y_2dB[tBB])
            print(Y_2d[t], Y_2dB[tBB])
            tBB = tBB + 1
            t = t + 1
        if np.array(df_listA[i][4]) == '3':
            X_2d[t] = np.array(X_2dC[tCC])
            Y_2d[t] = np.array(Y_2dC[tCC])
            print(Y_2d[t], Y_2dC[tCC])
            tCC = tCC + 1
            t = t + 1
        if np.array(df_listA[i][4]) == '4':
            X_2d[t] = np.array(X_2dD[tDD])
            Y_2d[t] = np.array(Y_2dD[tDD])
            print(Y_2d[t], Y_2dD[tDD])
            tDD = tDD + 1
            t = t + 1


def data_label(op):
    It = 0
    if op == 0:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = df_list[i][4]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = df_listA[i][4]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
        # Choose different target values (labels) for different tasks
    if op == 1:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = df_list[i][31]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = df_listA[i][31]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
    if op == 2:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = X_2d[i][0]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = X_2d[i][0]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
    if op == 3:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = X_2d[i][1]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = X_2d[i][1]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
    if op == 4:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = X_2d[i][2]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = X_2d[i][2]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
    if op == 5:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = X_2d[i][3]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = X_2d[i][3]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)
    if op == 6:
        for i in range(120):
            if i == 0 or i == 3 or i == 4 or i == 6 or i == 42 or i == 84 or i == 114:
                continue
            Y_2d[It] = df_list[i][32]
            print(Y_2d[It])
            It = It + 1
        print(It)
        for i in range(51):
            if i == 0 or i == 31 or i == 32:
                continue;
            Y_2d[It] = df_listA[i][32]
            print(np.array(Y_2d[It]))
            It = It + 1
        print(It)

def data_feature(op):
    global dfAGE_list
    global dfACC_list
    if op == 0:
        # The loss results of the regressor are imported, and the first round of training is not used
        print("dfAGE_array:",dfAGE_array)
        dfAGE_list = dfAGE_array.tolist()
        print("dfAGE_list:",dfAGE_list)
    else:
        dfACC_list = dfACC_array.tolist()
        print("dfACC_list:", dfACC_list)
