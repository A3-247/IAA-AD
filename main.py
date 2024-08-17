from data import data_init, data_label, data_feature
from predict import classification

epoch=1

if __name__=='__main__':
    data_init();
    for now in range(epoch):
        for i in range(7):
            data_label(i)
            if i==0:
                data_feature(i)
                classification(epoch)
            else:
                





