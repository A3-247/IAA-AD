import csv
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_regression
import data
from math import sqrt
from sklearn.metrics import classification_report

RMSE = np.array([[10000 for i in range(1024)] for j in range(6)], dtype='float')
MSEEE = np.array([[10000 for i in range(1024)] for j in range(6)], dtype='float')
MAEEE = np.array([[10000 for i in range(1024)] for j in range(6)], dtype='float')
R222 = np.array([[10000 for i in range(1024)] for j in range(6)], dtype='float')

FFF = np.array([[10000 for i in range(1024)] for j in range(6)])

X_2dA = [[0 for i in range(8)] for j in range(29)]
X_2dB = [[0 for i in range(8)] for j in range(49)]
X_2dC = [[0 for i in range(8)] for j in range(51)]
X_2dD = [[0 for i in range(8)] for j in range(32)]


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


def regression(epoch, num):
    torch.set_default_tensor_type(torch.DoubleTensor)
    Best_rmse = 10000
    Best_flag = 10000
    Best_RE = 0
    Best_F1 = 0
    Best_FLAG = 0
    LOW_ACC = 1.1
    LOW_RE = 1.1
    LOW_F1 = 1.1
    LOW_FLAG = 0
    AVG_ACC = 0
    DF = 0;
    DB = 0;
    DV = 0;
    DM = 0;
    mse, mae, r2 = np.array(0, dtype='float'), np.array(0, dtype='float'), np.array(0, dtype='float')
    global RMSE
    global MSEEE
    global MAEEE
    global R222
    for iii in range(100, 700, 1):
        XX = np.array(data.XXX_2d, dtype=float)
        YY = np.array(data.Y_2d, dtype=float)
        selector = SelectKBest(f_regression, k=iii)
        X_new = selector.fit_transform(XX, YY)
        ZES = 0
        if epoch == 0:
            if num == 1 or num == 6:
                ZES = 8
            else:
                ZES = 7
        else:
            if num == 1 or num == 6:
                ZES = 9
            else:
                ZES = 8
        X_newA = [[0 for i in range(iii + ZES)] for j in range(161)]
        for kkk in range(50):
            model = SelfAttention(161, iii + ZES, iii + ZES)
            Feature = np.zeros((iii + ZES, 161))
            for i in range(161):
                for j in range(iii):
                    X_newA[i][j] = X_new[i][j]
            for i in range(161):
                t = iii
                if epoch == 0:
                    for j in range(ZES):
                        if 2 <= num <= 5:
                            if j == num - 2:
                                continue
                        X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                        t = t + 1
                else:
                    for j in range(ZES - 1):
                        if 2 <= num <= 5:
                            if j == num - 2:
                                continue
                        X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                        t = t + 1
                if epoch != 0:
                    X_newA[i][t] = np.array(data.dfACC_list[0][i], dtype=float)
            for i in range(iii + ZES):
                for j in range(161):
                    Feature[i][j] = np.array(X_newA[j][i])
            Feature_tensor = torch.from_numpy(Feature)
            YQ1 = model(Feature_tensor)
            tagAA = [0 for i in range(iii + ZES)]
            for i in range(iii + ZES):
                tagAA[i] = sum(YQ1[i]).detach().numpy()
            X_newAtt = [[0 for i in range(iii + ZES)] for j in range(161)]
            for i in range(161):
                for j in range(iii + ZES):
                    X_newAtt[i][j] = X_newA[i][j] * tagAA[j]
            X_train = [[0 for i in range(iii + ZES)] for j in range(138)]
            X_test = [[0 for i in range(iii + ZES)] for j in range(23)]
            Y_train = np.array([0 for i in range(138)], dtype='float')
            Y_test = np.array([0 for i in range(23)], dtype='float')
            ACC = 0;
            cnt = 0;
            for i in range(7):
                u = 0;
                v = 0;
                for j in range(161):
                    if j % 7 == i:
                        X_test[u] = X_newAtt[j]
                        Y_test[u] = YY[j]
                        u = u + 1
                    else:
                        X_train[v] = X_newAtt[j]
                        Y_train[v] = YY[j]
                        v = v + 1;
                clf = svm.SVR(kernel='linear', C=0.095)
                predict_test = clf.fit(X_train, Y_train.ravel())
                score_test = clf.score(X_test, Y_test)
                score_train = clf.score(X_train, Y_train)
                predict_test = clf.predict(X_test)
                mse = mse + mean_squared_error(Y_test, predict_test)
                mae = mae + mean_absolute_error(Y_test, predict_test)
                r2 = r2 + r2_score(Y_test, predict_test)
                cnt = cnt + 1
            mse = mse / cnt
            rmse = mse ** 0.5
            mae = mae / cnt
            r2 = r2 / cnt
            print("epoch:", iii, "kkk", kkk, f'MSE: {mse}', f'RMSE: {rmse}', f'MAE: {mae}', f'R^2: {r2}')
            if rmse < RMSE[num-1][iii]:
                MSEEE[num-1][iii] = mse
                RMSE[num-1][iii] = rmse
                MAEEE[num-1][iii] = mae
                R222[num-1][iii] = r2
                if epoch==0:
                    FFF[num-1][iii]=1
                else:
                    FFF[num-1][iii]=0
                PATH = "./at_RE" + str(num)+ "_" + str(iii) + ".pth"
                torch.save(model.state_dict(), PATH)
        print("epoch:", iii, f'MSE: {MSEEE[num-1][iii]}', f'RMSE: {RMSE[num-1][iii]}', f'MAE: {MAEEE[num-1][iii]}', f'R^2: {R222[num-1][iii]}')
        if RMSE[num-1][iii] < Best_rmse:
            Best_rmse = RMSE[num-1][iii]
            Best_flag=iii
    print(Best_rmse)
    return Best_flag

def lossR(epoch,YR,op):
    torch.set_default_tensor_type(torch.DoubleTensor)
    mse, mae, r2 = np.array(0, dtype='float'), np.array(0, dtype='float'), np.array(0, dtype='float')
    selector = SelectKBest(f_regression, k=YR)
    XX = np.array(data.XXX_2d, dtype=float)
    YY = np.array(data.Y_2d, dtype=float)
    X_new = selector.fit_transform(XX, YY)
    if FFF[op-1][YR] == 1:
        if op == 1 or op == 6:
            ZES = 8
        else:
            ZES = 7
    else:
        if op == 1 or op == 6:
            ZES = 9
        else:
            ZES = 8

    X_newA = [[0 for i in range(YR+ZES)] for j in range(161)]
    PRED = [0 for i in range(161)]
    model = SelfAttention(161, YR+ZES, YR+ZES)
    print(model)
    Feature = np.zeros((YR+ZES, 161))
    for i in range(161):
        for j in range(YR):
            X_newA[i][j] = X_new[i][j]

    for i in range(161):
        t = YR
        if epoch == 0:
            for j in range(ZES):
                if 2 <= op <= 5:
                    if j == op - 2:
                        continue
                X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                t = t + 1
        else:
            for j in range(ZES - 1):
                if 2 <= op <= 5:
                    if j == op - 2:
                        continue
                X_newA[i][t] = np.array(data.X_2d[i][j], dtype=float)
                t = t + 1
            X_newA[i][t] = np.array(data.dfACC_array[0][i], dtype=float)

    for i in range(YR+ZES):
        for j in range(161):
            Feature[i][j] = np.array(X_newA[j][i])

    Feature_tensor = torch.from_numpy(Feature)
    PATH = "./at_RE" + str(op) + "_" + str(YR) + ".pth"
    save_model = torch.load(PATH)
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    YQ1 = model(Feature_tensor)
    tagAA = [0 for i in range(YR+ZES)]
    for i in range(YR+ZES):
        tagAA[i] = sum(YQ1[i]).detach().numpy()
    X_newAtt = [[0 for i in range(YR+ZES)] for j in range(161)]

    for i in range(161):
        for j in range(YR+ZES):
            X_newAtt[i][j] = X_newA[i][j] * tagAA[j]

    X_train = [[0 for i in range(YR+ZES)] for j in range(138)]
    X_test = [[0 for i in range(YR+ZES)] for j in range(23)]
    Y_train = np.array([0 for i in range(138)], dtype='float')
    Y_test = np.array([0 for i in range(23)], dtype='float')

    ACC = 0;
    cnt = 0;
    for i in range(7):
        u = 0;
        v = 0;
        for j in range(161):
            if j % 7 == i:
                X_test[u] = X_newAtt[j]
                Y_test[u] = YY[j]
                u = u + 1
            else:
                X_train[v] = X_newAtt[j]
                Y_train[v] = YY[j]
                v = v + 1;
        clf = svm.SVR(kernel='linear', C=0.095)
        print(Y_test)
        print(Y_train)
        predict_test = clf.fit(X_train, Y_train.ravel())
        score_test = clf.score(X_test, Y_test)
        score_train = clf.score(X_train, Y_train)
        predict_test = clf.predict(X_test)
        print("SVM Test Score : %.4g" % (score_test))
        print("SVM Train Score : %.4g" % (score_train))
        mse = mse + mean_squared_error(Y_test, predict_test)
        mae = mae + mean_absolute_error(Y_test, predict_test)
        r2 = r2 + r2_score(Y_test, predict_test)
        cnt = cnt + 1
        for ii in range(len(predict_test)):
            PRED[ii * 7 + i] = predict_test[ii]
    mse = mse / cnt
    rmse = mse ** 0.5
    mae = mae / cnt
    r2 = r2 / cnt
    print(f'MSE: {mse}', f'RMSE: {rmse}', f'MAE: {mae}', f'R^2: {r2}')
    Y_sub = [0 for i in range(161)]
    for i in range(161):
        Y_sub[i] = YY[i] - PRED[i]
    print(Y_sub)
    lists = [Y_sub]
    print(Y_sub)
    print(lists)
    with open('./R.txt', 'a') as f:
        f.write("epoch:"+str(epoch)+" YR:"+str(YR)+" op:"+str(op)+" rmse:"+str(rmse)+" mae:"+str(mae)+" r2:"+str(r2)+"\n")
        f.close()
    if op==1:
        with open("./R_loss.csv", "w", encoding="utf-8", newline='') as f:
            csv_writer = csv.writer(f)
            # name = ['R_loss']
            # csv_writer.writerow(name)
            # csv_writer.writerow(lists)
            for LIST in lists:
                csv_writer.writerow(LIST)
            f.close()
    else:
        with open("./R_loss.csv", "a", encoding="utf-8",newline='') as f:
            csv_writer = csv.writer(f)
            # name = ['Y_loss']
            # csv_writer.writerow(name)
            # csv_writer.writerow(lists)
            for LIST in lists:
                csv_writer.writerow(LIST)
            f.close()
