# import readbin
# import numpy as np
# "matplotlib.use('agg')"
# import matplotlib.pyplot as plt
# from sklearn import svm


# filename = "C:\\Users\\Hong\\Desktop\\u=3001031685,395006647&fm=26&gp=0.jpg"
# data = readbin.trainbin(filename,(920,20,960,157),(980,70,983,177))
# train_data = data[:,0:3]
# train_traget = data[:,3]
# clf = svm.SVC(kernel = 'poly')
# clf.fit(train_data, train_traget, sample_weight = None)
# test_data = readbin.readjpg(filename)
# result = clf.predict(test_data).reshape(348,500)
# readbin.draw_cls(result,2)
# from tensorflow.examples.turorials.mnist import input_data


# mnist = input_data.read_data_sets('.',one_hot = True)
# print(mnist)

import numpy as np
import clsfi_ae

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
data = np.load("C:\\Users\\Administrator\\Desktop\\samp.npy")
num = clsfi_ae.clsfi_ae(sampelist,data)
print(num)