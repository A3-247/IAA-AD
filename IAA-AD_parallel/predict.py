import pandas as pd
import pymp
from sklearn.feature_selection import SelectKBest, f_regression

import data

from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
from torch import tensor
import csv

AAA = [0 for i in range(600)]



def categorical_cross_entropy(y_true, y_pred):
    # 避免出现概率为0的情况，加上一个小的偏移量
    epsilon = 1e-7
    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred + epsilon))
    return loss


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        batch, dim_q = x.shape
        n = 1
        assert dim_q == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = torch.mm(q, k.transpose(0, 1)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.mm(dist, v)
        return att


def classification(epoch):
    torch.set_default_tensor_type(torch.DoubleTensor)
    Best_ACC = 0
    Best_FLAG = 0
    LOW_ACC = 1.1
    LOW_FLAG = 0
    AVG_ACC = 0
    DF = 0
    DB = 0
    DV = 0
    DM = 0
    ACCNOW = [0 for i in range(825)]
    TT = 0
    for kkk in range(4):
        for i in range(100, 700):
            if i % 4 == kkk:
                AAA[TT] = i
                TT = TT + 1
    if epoch == 0:
        with pymp.Parallel(4) as P:
            for index in P.range(0, 600, 1):
                iii=AAA[index]
                XX = np.array(data.XXX_2d, dtype=float)
                YY = np.array(data.Y_2d, dtype=float)
                # print(YY)
                selector = SelectKBest(f_regression, k=iii)
                X_new = selector.fit_transform(XX, YY)
                X_newA = [[0 for i in range(iii + 7)] for j in range(161)]
                for kkk in range(50):
                    model = SelfAttention(161, iii + 7, iii + 7)
                    Feature = np.zeros((iii + 7, 161))
                    for i in range(161):
                        for j in range(iii):
                            X_newA[i][j] = X_new[i][j]
                    for i in range(161):
                        t = iii
                        for j in range(7):
                            X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                            t = t + 1
                    for i in range(iii + 7):
                        for j in range(161):
                            Feature[i][j] = np.array(X_newA[j][i])
                    Feature_tensor = torch.from_numpy(Feature)
                    # with P.lock:
                    YQ1 = model(Feature_tensor)
                    tagAA = [0 for i in range(iii + 7)]
                    for i in range(iii + 7):
                        tagAA[i] = sum(YQ1[i]).detach().numpy()
                    X_newAtt = [[0 for i in range(iii + 7)] for j in range(161)]
                    for i in range(161):
                        for j in range(iii + 7):
                            X_newAtt[i][j] = X_newA[i][j] * tagAA[j]
                    X_train = [[0 for i in range(iii + 7)] for j in range(138)]
                    X_test = [[0 for i in range(iii + 7)] for j in range(23)]
                    Y_train = [0 for i in range(138)]
                    Y_test = [0 for i in range(23)]
                    ACC = 0;
                    cnt = 0;
                    for i in range(7):
                        u = 0;
                        v = 0;
                        for j in range(161):
                            if j % 7 == i:
                                X_test[u] = X_newAtt[j]
                                Y_test[u] = YY[j]
                                u = u + 1;
                            else:
                                X_train[v] = X_newAtt[j]
                                Y_train[v] = YY[j]
                                v = v + 1;
                        with P.lock:
                            clf_linear = svm.SVC(kernel='linear', C=0.095)
                        clf_linear.fit(X_train, Y_train)
                        score_linear_test = clf_linear.score(X_test, Y_test)
                        score_linear_train = clf_linear.score(X_train, Y_train)
                        predict_test = clf_linear.predict(X_test)
                        ACC = ACC + score_linear_test;
                        cnt = cnt + 1;
                    ACC = ACC / cnt
                    AVG_ACC = AVG_ACC + ACC
                    if ACC > Best_ACC:
                        Best_FLAG = iii
                        Best_ACC = ACC
                    if ACC < LOW_ACC:
                        LOW_FLAG = iii
                        LOW_ACC = ACC
                    print(iii, kkk, ACC)
                    if ACCNOW[iii] < ACC:
                        ACCNOW[iii] = ACC
                        PATH = "./ARZE" + str(iii) + ".pth"
                        torch.save(model.state_dict(), PATH)
                print(iii, ACCNOW[iii])
    else:
        with pymp.Parallel(4) as P:
            for index in P.range(0, 600, 1):
                iii = AAA[index]
                XX = np.array(data.XXX_2d, dtype=float)
                YY = np.array(data.Y_2d, dtype=float)
                selector = SelectKBest(f_regression, k=iii)
                X_new = selector.fit_transform(XX, YY)
                X_newA = [[0 for i in range(iii + 13)] for j in range(161)]
                for kkk in range(50):
                    model = SelfAttention(161, iii + 13, iii + 13)
                    Feature = np.zeros((iii + 13, 161))
                    for i in range(161):
                        for j in range(iii):
                            X_newA[i][j] = X_new[i][j]
                    for i in range(161):
                        t = iii
                        for j in range(7):
                            X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                            t = t + 1
                        for j in range(6):
                            X_newA[i][t] = np.array(data.dfAGE_list[j][i], dtype=float)
                            t = t + 1
                    for i in range(iii + 13):
                        for j in range(161):
                            Feature[i][j] = np.array(X_newA[j][i])
                    Feature_tensor = torch.from_numpy(Feature)
                    # with P.lock:
                    YQ1 = model(Feature_tensor)
                    tagAA = [0 for i in range(iii + 13)]
                    for i in range(iii + 13):
                        tagAA[i] = sum(YQ1[i]).detach().numpy()
                    X_newAtt = [[0 for i in range(iii + 13)] for j in range(161)]
                    for i in range(161):
                        for j in range(iii + 13):
                            X_newAtt[i][j] = X_newA[i][j] * tagAA[j]
                    X_train = [[0 for i in range(iii + 13)] for j in range(138)]
                    X_test = [[0 for i in range(iii + 13)] for j in range(23)]
                    Y_train = [0 for i in range(138)]
                    Y_test = [0 for i in range(23)]
                    ACC = 0;
                    cnt = 0;
                    for i in range(7):
                        u = 0;
                        v = 0;
                        for j in range(161):
                            if j % 7 == i:
                                X_test[u] = X_newAtt[j]
                                Y_test[u] = YY[j]
                                u = u + 1;
                            else:
                                X_train[v] = X_newAtt[j]
                                Y_train[v] = YY[j]
                                v = v + 1;
                        with P.lock:
                            clf_linear = svm.SVC(kernel='linear', C=0.095)
                        clf_linear.fit(X_train, Y_train)
                        score_linear_test = clf_linear.score(X_test, Y_test)
                        score_linear_train = clf_linear.score(X_train, Y_train)
                        predict_test = clf_linear.predict(X_test)
                        ACC = ACC + score_linear_test;
                        cnt = cnt + 1;
                    ACC = ACC / cnt
                    AVG_ACC = AVG_ACC + ACC
                    if ACC > Best_ACC:
                        Best_FLAG = iii
                        Best_ACC = ACC
                    if ACC < LOW_ACC:
                        LOW_FLAG = iii
                        LOW_ACC = ACC
                    print(iii, kkk, ACC)
                    if ACCNOW[iii] < ACC:
                        ACCNOW[iii] = ACC
                        PATH = "./ARZE" + str(iii) + ".pth"
                        torch.save(model.state_dict(), PATH)
                print(iii, ACCNOW[iii])
    print("BEST:")
    print(Best_FLAG, Best_ACC)
    print(LOW_FLAG, LOW_ACC)
    return Best_FLAG


def lossC(epoch, YC):
    import torch
    from torch import tensor
    torch.set_default_tensor_type(torch.DoubleTensor)
    PRED = [0 for i in range(161)]
    SOFTMAX = []
    Best_ACC = 0
    Best_RE = 0
    Best_F1 = 0
    Best_FLAG = 0
    LOW_ACC = 1.1
    LOW_RE = 1.1
    LOW_F1 = 1.1
    LOW_FLAG = 0
    AVG_ACC = 0
    AC_F = 0;
    AC_B = 0;
    AC_V = 0;
    AC_M = 0;
    all_F = 0;
    all_B = 0;
    all_V = 0;
    all_M = 0;
    DF = 0;
    DB = 0;
    DV = 0;
    DM = 0;
    # print("YC:",YC,epoch)
    selector = SelectKBest(f_regression, k=YC)
    XX = np.array(data.XXX_2d, dtype=float)
    YY = np.array(data.Y_2d, dtype=float)
    X_new = selector.fit_transform(XX, YY)
    if epoch == 0:
        X_newA = [[0 for i in range(YC + 7)] for j in range(161)]
        model = SelfAttention(161, YC + 7, YC + 7)
        save_model = torch.load("./ARZE" + str(YC) + ".pth")
        model_dict = model.state_dict()

        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        Feature = np.zeros((YC + 7, 161))
        for i in range(161):
            for j in range(YC):
                X_newA[i][j] = X_new[i][j]
        for i in range(161):
            t = YC
            for j in range(7):
                X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                t = t + 1
        for i in range(YC + 7):
            for j in range(161):
                Feature[i][j] = np.array(X_newA[j][i])
        Feature_tensor = torch.from_numpy(Feature)
        YQ1 = model(Feature_tensor)
        tagAA = [0 for i in range(YC + 7)]
        for i in range(YC + 7):
            tagAA[i] = sum(YQ1[i]).detach().numpy()
        X_newAtt = [[0 for i in range(YC + 7)] for j in range(161)]
        for i in range(161):
            for j in range(YC + 7):
                X_newAtt[i][j] = X_newA[i][j] * tagAA[j]
        X_train = [[0 for i in range(YC + 7)] for j in range(138)]
        X_test = [[0 for i in range(YC + 7)] for j in range(23)]
        Y_train = [0 for i in range(138)]
        Y_test = [0 for i in range(23)]
        ACC = 0;
        cnt = 0;
        for i in range(7):
            u = 0;
            v = 0;
            for j in range(161):
                if j % 7 == i:
                    X_test[u] = X_newAtt[j]
                    Y_test[u] = YY[j]
                    if Y_test[u] == 1:
                        DF = DF + 1;
                    if Y_test[u] == 2:
                        DB = DB + 1;
                    if Y_test[u] == 3:
                        DV = DV + 1;
                    if Y_test[u] == 4:
                        DM = DM + 1;
                    u = u + 1;
                else:
                    X_train[v] = X_newAtt[j]
                    Y_train[v] = YY[j]
                    v = v + 1;
            clf_linear = svm.SVC(kernel='linear', C=0.095, probability=True)
            clf_linear.fit(X_train, Y_train)
            score_linear_test = clf_linear.score(X_test, Y_test)
            score_linear_train = clf_linear.score(X_train, Y_train)
            predict_test = clf_linear.predict(X_test)
            print("SVM Test  Accuracy : %.4g" % (score_linear_test))
            print("SVM Train  Accuracy : %.4g" % (score_linear_train))
            ACC = ACC + score_linear_test;
            cnt = cnt + 1;
            print(predict_test)
            for iii in range(len(predict_test)):
                PRED[iii * 7 + i] = predict_test[iii]
            for ii in range(len(Y_test)):
                if predict_test[ii] == 1:
                    all_F = all_F + 1;
                    if Y_test[ii] == 1:
                        AC_F = AC_F + 1
                if predict_test[ii] == 2:
                    all_B = all_B + 1;
                    if Y_test[ii] == 2:
                        AC_B = AC_B + 1
                if predict_test[ii] == 3:
                    all_V = all_V + 1;
                    if Y_test[ii] == 3:
                        AC_V = AC_V + 1
                if predict_test[ii] == 4:
                    all_M = all_M + 1;
                    if Y_test[ii] == 4:
                        AC_M = AC_M + 1
            SOFTMAX.append(clf_linear.predict_proba(X_test))
        ACC = ACC / cnt
        print(ACC)
    else:
        X_newA = [[0 for i in range(YC + 13)] for j in range(161)]
        model = SelfAttention(161, YC + 13, YC + 13)
        save_model = torch.load("./ARZE" + str(YC) + ".pth")
        model_dict = model.state_dict()

        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        Feature = np.zeros((YC + 13, 161))
        for i in range(161):
            for j in range(YC):
                X_newA[i][j] = X_new[i][j]
        for i in range(161):
            t = YC
            for j in range(7):
                X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                t = t + 1
            for j in range(6):
                X_newA[i][t] = np.array(data.dfAGE_list[j][i], dtype=float)
                t = t + 1
        for i in range(YC + 13):
            for j in range(161):
                Feature[i][j] = np.array(X_newA[j][i])
        Feature_tensor = torch.from_numpy(Feature)
        YQ1 = model(Feature_tensor)
        tagAA = [0 for i in range(YC + 13)]
        for i in range(YC + 13):
            tagAA[i] = sum(YQ1[i]).detach().numpy()
        X_newAtt = [[0 for i in range(YC + 13)] for j in range(161)]
        for i in range(161):
            for j in range(YC + 13):
                X_newAtt[i][j] = X_newA[i][j] * tagAA[j]
        X_train = [[0 for i in range(YC + 13)] for j in range(138)]
        X_test = [[0 for i in range(YC + 13)] for j in range(23)]
        Y_train = [0 for i in range(138)]
        Y_test = [0 for i in range(23)]
        ACC = 0;
        cnt = 0;
        for i in range(7):
            u = 0;
            v = 0;
            for j in range(161):
                if j % 7 == i:
                    X_test[u] = X_newAtt[j]
                    Y_test[u] = YY[j]
                    if Y_test[u] == 1:
                        DF = DF + 1;
                    if Y_test[u] == 2:
                        DB = DB + 1;
                    if Y_test[u] == 3:
                        DV = DV + 1;
                    if Y_test[u] == 4:
                        DM = DM + 1;
                    u = u + 1;
                else:
                    X_train[v] = X_newAtt[j]
                    Y_train[v] = YY[j]
                    v = v + 1;
            clf_linear = svm.SVC(kernel='linear', C=0.095, probability=True)
            clf_linear.fit(X_train, Y_train)
            score_linear_test = clf_linear.score(X_test, Y_test)
            score_linear_train = clf_linear.score(X_train, Y_train)
            predict_test = clf_linear.predict(X_test)
            print("SVM Test  Accuracy : %.4g" % (score_linear_test))
            print("SVM Train  Accuracy : %.4g" % (score_linear_train))
            ACC = ACC + score_linear_test;
            cnt = cnt + 1;
            print(predict_test)
            for iii in range(len(predict_test)):
                PRED[iii * 7 + i] = predict_test[iii]
            for ii in range(len(Y_test)):
                if predict_test[ii] == 1:
                    all_F = all_F + 1;
                    if Y_test[ii] == 1:
                        AC_F = AC_F + 1
                if predict_test[ii] == 2:
                    all_B = all_B + 1;
                    if Y_test[ii] == 2:
                        AC_B = AC_B + 1
                if predict_test[ii] == 3:
                    all_V = all_V + 1;
                    if Y_test[ii] == 3:
                        AC_V = AC_V + 1
                if predict_test[ii] == 4:
                    all_M = all_M + 1;
                    if Y_test[ii] == 4:
                        AC_M = AC_M + 1
            SOFTMAX.append(clf_linear.predict_proba(X_test))
        ACC = ACC / cnt
        print(ACC)

    from sklearn.metrics import classification_report
    target_names = ['AD', 'CN', 'EMCI', 'LMCI']
    print(classification_report(PRED, YY, target_names=target_names, digits=4))

    X_hot = np.array([[0 for i in range(4)] for j in range(161)], dtype='float')
    for i in range(7):
        for j in range(23):
            for k in range(4):
                X_hot[j * 7 + i][k] = SOFTMAX[i][j][k]

    YY = np.array(YY, dtype='int')
    for i in range(161):
        YY[i] = YY[i] - 1
    Y_hot = np.eye(4)[YY]

    Y_loss = [0 for i in range(161)]
    for i in range(161):
        Y_loss[i] = categorical_cross_entropy(Y_hot[i], X_hot[i])

    # Y_loss = pd.DataFrame(Y_loss)

    lists = [Y_loss]
    print(Y_loss)
    print(lists)
    # lists = list(zip(*lists))
    # print("list:",lists)

    for LIST in lists:
        print(LIST)
    with open('./C.txt', 'a') as f:
        f.write("epoch:" + str(epoch) + " YC:" + str(YC) + " ACC:" + str(ACC)+"\n")
        f.close()
    with open("./C_loss.csv", "w", encoding="utf-8", newline='') as f:
        csv_writer = csv.writer(f)
        # name = ['C_loss']
        # csv_writer.writerow(name)
        # csv_writer.writerow(lists)
        for LIST in lists:
            csv_writer.writerow(LIST)
        f.close()
