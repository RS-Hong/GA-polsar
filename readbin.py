import numpy as np
import cv2
import struct
import os
import re
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


def all_path(dirname):

    result = []#所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            if os.path.splitext(apath)[1] == '.bin':                    
                result.append(apath)
    return result


def readbin(lines,samples,filename):
    '''读取二进制bin文件'''
    t3 = np.zeros([lines, samples], dtype=float)
    with open(filename, "rb") as f:
        for i in range(lines):
            data = f.read(4 * samples)
            t3[i, ...] = struct.unpack(str(samples) + 'f', data)
    return t3

    
def readbin2(lines,samples,filename):
    t3_real = np.zeros([lines,samples],dtype = float)
    t3_imag = np.zeros([lines,samples],dtype = float)
    real_basename = os.path.basename(filename)
    imag_basename = re.match(r'T\d+',real_basename).group()
    imag_basename = imag_basename + "_imag.bin"
    dirname = os.path.dirname(filename)
    imag_name = os.path.join(dirname,imag_basename)
    with open(filename, "rb") as f:
        for i in range(lines):
            data = f.read(4 * samples)
            t3_real[i, ...] = struct.unpack(str(samples) + 'f', data)
    with open(imag_name,"rb") as f:
        for i in range(lines):
            data = f.read(4 * samples)
            t3_imag[i, ...] = struct.unpack(str(samples) + 'f', data)
    mod = np.sqrt(t3_real **2 + t3_imag **2)
    return mod


def readallbin(directory,x,y):   
    #dataset 的维度需要手动输入
    allpath = all_path(directory)   
    allpath.remove(os.path.join(directory,'mask_valid_pixels.bin'))
    print(allpath)
    #构造数据集，维度为所有特征个数 其中T12 T13 T23 C12 C13 C23由两个文件构成
    trait_pool = []
    dataset = np.zeros((x,y,13),dtype = 'float')
    i = 0
    for filename in allpath:
        if os.path.basename(filename) in ['T11.bin' , "T22.bin" , "T33.bin"]:
            trait_name = re.match(r"T\d+",os.path.basename(filename))[0]
            trait_pool.append(trait_name)
            dataset[:,:,i] = readbin(x,y,filename)
            i += 1
        elif os.path.basename(filename) in ['T12_real.bin' , 'T13_real.bin' , 'T23_real.bin']:
            trait_name = re.match(r"T\d+",os.path.basename(filename))[0]
            trait_pool.append(trait_name)
            dataset[:,:,i] = readbin2(x,y,filename)
            i += 1
        elif os.path.basename(filename) in ['alpha.bin',
                                            'anisotroy.bin',
                                            'entropy.bin',
                                            'lambda.bin',
                                            'Yamaguchi3_Dbl.bin', 
                                            'Yamaguchi3_Odd.bin', 
                                            'Yamaguchi3_Vol.bin']:
            trait_name = os.path.basename(filename).split('.')[0]
            trait_pool.append(trait_name)
            dataset[:,:,i] = readbin(x,y,filename)
            i += 1
    return dataset,trait_pool


def trainbin(data,samplelist):
    '''根据data和samplelist得到样本数据集'''
    #samplelist形式   [(),(),()]
    k = int(len(samplelist))
    traitnum = range(data.shape[2])  # 特征个数
    calnum = 0  #计数器
    for sample in samplelist:
        nums = int(len(sample)/4)   #该类别样本个数
        for i in range(nums):
            x1, y1, x2, y2= sample[4*i], sample[4*i+1], sample[4*i+2], sample[4*i+3]
            train_data = np.zeros(((x2-x1)*(y2-y1),len(traitnum) + 1),dtype = 'float')
            if calnum == 0:
                for trait in traitnum:
                    train_data[:, traitnum.index(trait)] = np.array(data[x1:x2, y1:y2, trait]).flatten().T
                #19.10.22 修改为类别标记由1开始
                train_data[:,len(traitnum)] = samplelist.index(sample) + 1 
                traindata = train_data
                calnum += 1
            else:
                for trait in traitnum:
                    train_data[:, traitnum.index(trait)] = np.array(data[x1:x2, y1:y2, trait]).flatten().T
                train_data[:,len(traitnum)] = samplelist.index(sample) + 1
                traindata = np.concatenate([traindata,train_data])
    return traindata
    # k = int(len(lt) / 2)
    #   #样本类别个数
    # traitnum = range(data.shape[2])  #特征个数
    # for i in range(k):
    #     if i == 0:
    #         x1,y1 = lt[2*i],lt[2*i+1]
    #         x2,y2 = rb[2*i],rb[2*i+1]
    #         train_data = np.zeros(((x2-x1) * (y2-y1), len(traitnum) + 1),dtype = 'float')
    #         for samp in traitnum:
    #             train_data[:,traitnum.index(samp)] = np.array(data[x1:x2,y1:y2,samp]).flatten().T
    #         train_data[:,len(traitnum)] = i
    #         traindata = train_data
    #     else:
    #         x1,y1 = lt[2*i],lt[2*i+1]
    #         x2,y2 = rb[2*i],rb[2*i+1]
    #         train_data = np.zeros(((x2-x1) * (y2-y1), len(traitnum) + 1),dtype = 'float')
    #         for samp in traitnum:
    #             train_data[:,traitnum.index(samp)] = np.array(data[x1:x2,y1:y2,samp]).flatten().T
    #         train_data[:,len(traitnum)] = i
    #         traindata = np.concatenate([traindata,train_data])
    # return traindata



def draw_cls(cls_data,k):
    '''绘制类别彩色图像
    cls_data应是由类别编号组成的二维矩阵,k为类别数'''
    print(cls_data.shape)
    x,y = cls_data.shape
    picture_cls = np.zeros((x*y,3),dtype = 'uint8')
    color_pool = [[0,0,139],
                  [144,238,144],
                  [224,255,255],
                  [139,0,139],
                  [139,0,0],
                  [238,174,238],
                  [255,127,0],
                  [255,255,0],
                  [0,255,255],
                  [34,139,34],
                  [0,100,0],
                  [0,238,238],
                  [139,69,19],
                  [238,99,99],
                  [255,0,255]]
    for i in range(k):
        picture_cls[np.nonzero(cls_data.flatten() == i + 1),0] = color_pool[i][0]
        picture_cls[np.nonzero(cls_data.flatten() == i + 1),1] = color_pool[i][1]
        picture_cls[np.nonzero(cls_data.flatten() == i + 1),2] = color_pool[i][2]   
    b = picture_cls[:,0].reshape(x,y)
    g = picture_cls[:,1].reshape(x,y)
    r = picture_cls[:,2].reshape(x,y)
    res = np.zeros((x, y, 3), dtype='uint8')
    res[:, :, 0] = r
    res[:, :, 1] = g
    res[:, :, 2] = b
    cv2.imshow('hehe', res)
    cv2.waitKey(0)


def draw_fit(fit_pool):
    '''绘制适应度函数随迭代次数增加折线图'''
    y = fit_pool
    x = [i+1 for i in list(range(len(fit_pool)))]
    plt.plot(x,y,marker = 'o',mec = 'r',mfc = 'w',label = u'适应度函数折线图')
    plt.legend()
    plt.xticks(x)
    plt.margins(0)
    plt.subplots_adjust(bottom = 0.15)
    plt.ylabel(u'适应度函数值')
    plt.xlabel(u'迭代次数')
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.show()


if __name__ == '__main__':
    filename = "C:\\Users\\Hong\\Desktop\\开题PPT\\AIRSAR_Flevoland_LEE\\T3"
    dataset,trait_pool = readallbin(filename,750,1024)
    print(dataset[500,500,:])
    print(trait_pool)

    # filename = "C:\\Users\\Hong\\Desktop\\北京电子所\\temp\\T3\\T11.bin"
    # t3 = readbin(1279,1024,filename)
    # print(t3[500,500])