from sklearn.feature_selection import SelectKBest, f_regression

import data

from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
from torch import tensor


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
    if epoch == 0:
        for iii in range(100, 700, 1):
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
                        X_newA[i][t] = np.array(data.dfAGE_list[i][j], dtype=float)
                        t = t + 1
                for i in range(iii + 13):
                    for j in range(161):
                        Feature[i][j] = np.array(X_newA[j][i])
                Feature_tensor = torch.from_numpy(Feature)
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
                        if (j % 7 == i):
                            X_test[u] = X_newAtt[j]
                            Y_test[u] = YY[j]
                            u = u + 1;
                        else:
                            X_train[v] = X_newAtt[j]
                            Y_train[v] = YY[j]
                            v = v + 1;
                    clf_linear = svm.SVC(kernel='linear', C=0.095)
                    clf_linear.fit(X_train, Y_train)
                    score_linear_test = clf_linear.score(X_test, Y_test)
                    score_linear_train = clf_linear.score(X_train, Y_train)
                    predict_test = clf_linear.predict(X_test)
                    #         print("SVM Test  Accuracy : %.4g" % (score_linear_test))
                    #         print("SVM Train  Accuracy : %.4g" % (score_linear_train))
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
                if (ACCNOW[iii] < ACC):
                    ACCNOW[iii] = ACC
                    PATH = "./ARZE" + str(iii) + "G1.pth"
                    torch.save(model.state_dict(), PATH)
                print(iii, ACCNOW[iii])
    else:
        for iii in range(100, 700, 1):
            XX = np.array(data.XXX_2d, dtype=float)
            YY = np.array(data.Y_2d, dtype=float)
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
                    for j in range(6):
                        X_newA[i][t] = np.array(data.dfAGE_list[i][j], dtype=float)
                        t = t + 1
                for i in range(iii + 7):
                    for j in range(161):
                        Feature[i][j] = np.array(X_newA[j][i])
                Feature_tensor = torch.from_numpy(Feature)
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
                        if (j % 7 == i):
                            X_test[u] = X_newAtt[j]
                            Y_test[u] = YY[j]
                            u = u + 1;
                        else:
                            X_train[v] = X_newAtt[j]
                            Y_train[v] = YY[j]
                            v = v + 1;
                    clf_linear = svm.SVC(kernel='linear', C=0.095)
                    clf_linear.fit(X_train, Y_train)
                    score_linear_test = clf_linear.score(X_test, Y_test)
                    score_linear_train = clf_linear.score(X_train, Y_train)
                    predict_test = clf_linear.predict(X_test)
                    #         print("SVM Test  Accuracy : %.4g" % (score_linear_test))
                    #         print("SVM Train  Accuracy : %.4g" % (score_linear_train))
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
                if (ACCNOW[iii] < ACC):
                    ACCNOW[iii] = ACC
                    PATH = "./ARZE" + str(iii) + "G1.pth"
                    torch.save(model.state_dict(), PATH)
                print(iii, ACCNOW[iii])
    print("BEST:")
    print(Best_FLAG, Best_ACC)
    print(LOW_FLAG, LOW_ACC)
    return Best_FLAG

