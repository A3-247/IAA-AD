import csv

import numpy as np
import pandas as pd

import data
from data import data_init, data_label, data_feature
from predict import classification,lossC
from regress import regression,lossR

epoch=3
FLAG=1

if __name__=='__main__':
    data_init();

    for now in range(epoch):

        with open("./R_loss.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            print("reader:",reader)
            TT=0
            for line in reader:
                print(line)
                data.dfAGE_array[TT]=line
                print(data.dfAGE_array[TT],TT)
                TT=TT+1
            csvfile.close()
            print("TT:",TT)

        with open("./C_loss.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            print("reader:",reader)
            TT=0
            for line in reader:
                print(line)
                data.dfACC_array[TT]=line
                print(data.dfACC_array[TT],TT)
                TT=TT+1
            csvfile.close()
            print("TT:",TT)
        for i in range(7):
            data_label(i)
            if i==0:
                print("epoch:",now)
                if epoch!=0:
                    data_feature(i)
                YC=classification(now)
                print("epoch:",now)
                lossC(now,YC)
            else:
                print("I:",i)
                if epoch!=0:
                    data_feature(i)
                YR=regression(now,i)
                lossR(now,YR,i)