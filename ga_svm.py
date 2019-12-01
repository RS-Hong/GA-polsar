import clsfi_ae
import readbin
import numpy as np
import time
"matplotlib.use('agg')"
import matplotlib.pyplot as plt
from sklearn import svm
def msvm(dataset):
    sampelist = [(11,917,47,991,73,945,120,1020,219,970,247,1024,358,996,389,1020),
                 (148,957,173,976,489,900,518,924,546,922,571,947),
                 (321,778,341,809,382,783,471,820),
                 (545,767,574,810,667,772,688,811,249,203,274,260),
                 (394,621,417,706,489,695,521,738,391,562,417,590),
                 (660,691,702,703),
                 (560,278,588,317,439,346,451,444),
                 (393,349,418,433,395,277,424,304),
                 (556,359,587,433,674,369,703,414),
                 (667,297,705,308,337,209,369,229),
                 (669,492,697,579,721,482,741,523),
                 (551,488,583,546,716,703,739,744),
                 (297,195,319,278),
                 (308,469,319,532,279,529,288,555,266,612,286,633),
                 (223,650,246,681,261,699,287,719)]
    x, y, k = dataset.shape
    data = readbin.trainbin(dataset,sampelist)   #得到训练数据集
    train_data = data[:,0:data.shape[1] - 1]
    train_target = data[:,data.shape[1] - 1].astype(int)
    clf = svm.SVC(kernel = 'rbf',gamma=1.0)
    clf.fit(train_data, train_target, sample_weight = None)
    test_data = np.zeros((x * y,k),dtype= 'float')
    j = 0
    for i in range(k):
        test_data[:,j] = np.array(dataset[:,:,j]).flatten().T
        j +=1
    result = clf.predict(test_data).reshape(750,1024)
    # readbin.draw_cls(result,15)
    # np.save("C:\\Users\\Administrator\\Desktop\\samp.npy",result)
    fp, cfx_mx = clsfi_ae.clsfi_ae(sampelist,result)
    return fp, cfx_mx, result
if __name__ == '__main__':
    directory = "C:\\Users\\Administrator\\Desktop\\AIRSAR_Flevoland_LEE\\T3"
    dataset, trait_pool = readbin.readallbin(directory, 750, 1024)

    msvm(dataset)