import copy
import numpy as np
import random
import readbin
import cv2
#读取多维数据集
#转化为k-means处理的行样本格式
#确定初始聚类中心
#迭代计算新的聚类中心

def ini_center(x,k,dat):
    center_pool = []    #判定初始族心是否相同
    center = np.zeros((k,x), dtype = 'uint8')
    for i in range(k):
        ini_center = []
        for j in range(x):
            ini_center.append(np.random.randint(min(dat[j,...]), high = max(dat[j,...])))
        if ini_center not in center_pool:
            center_pool.append(ini_center)
            center[i,...] =ini_center 
    return center

def cal_eul(prep,cent):
    '''计算欧氏距离
    dat表示数据集，prep当前点数据，cent中心店数据'''
    return np.sqrt(sum(np.power(np.array(cent) - np.array(prep), 2)))


def cal_cluster(dat,center,dataclass,k):
    '''聚类'''
    y = dataclass.shape[0]
    for i in range(y):
        for j in range(k):
            prep = dat[...,i]
            cent = center[j,...]
            dist = cal_eul(prep,cent)
            if dataclass[i,0] == 0:
                dataclass[i,...] = dist,j
            elif dist < dataclass[i,0]:
                dataclass[i,...] = dist,j
    return dataclass

def km_jpg(dat,k):
    '''用于计算jpg图像的均值聚类算法
    x是维度    y是数据个数  k表示类别数'''
    [x,y] = dat.shape
    result = np.mat(np.zeros((3,y), dtype = 'uint8'))
    #记录每个数据点的族心的距离以及类别
    dataclass = np.mat(np.zeros((y,2),dtype = 'int'))
    # center = ini_center(x,k,dat)
    center = np.array([[247,247,246],[104,119,135],[105,119,135]])
    change = True
    while change:
        old_center = copy.copy(center)
        dataclass = cal_cluster(dat,center,dataclass,k)
        #计算新的聚类中心，更新聚类直到质心不在发生变化
        for cent in range(k):
            cent_data = dat[:,np.nonzero(dataclass[:,1].A == cent)[0]]
            center[cent,:] = np.mean(cent_data,axis = 1)
        print(np.sum(np.mat(center, dtype='int') - np.mat(old_center, dtype='int')))
        if np.sum(np.mat(center, dtype='int') - np.mat(old_center, dtype='int')) == 0:
            change = False
    for i in range(k):
        if i == 0:
            result[:,np.nonzero(dataclass[:,1].A == i)[0]] = [[255],[0],[0]]
        elif i == 1:
            result[:,np.nonzero(dataclass[:,1].A == i)[0]] = [[0],[255],[0]]
        elif i == 2:
            result[:,np.nonzero(dataclass[:,1].A == i)[0]] = [[0],[0],[255]]
    print(center)
    return result.A

def main():
    k = 3
    # path = "C:\\Users\\Hong\\Desktop\\聚类测试1.jpg"
    # "u = 876926361, 2082598989 & fm = 15 & gp = 0.jpg"
    jpg = cv2.imread("C:\\Users\\Hong\\Desktop\\ceshi2.jpg")
    [x, y, d] = jpg.shape
    b = jpg[:, :, 0].flatten()
    g = jpg[:, :, 1].flatten()
    r = jpg[:, :, 2].flatten()
    # 得到n行一维数组表示的数据集
    dat = np.array([r, g, b])
    # c_jpg = dat
    c_jpg = km_jpg(dat, k)
    b = c_jpg[0, :].reshape(x, y)
    g = c_jpg[1, :].reshape(x, y)
    r = c_jpg[2, :].reshape(x, y)
    res = np.zeros((x, y, d), dtype='uint8')
    res[:, :, 0] = r
    res[:, :, 1] = g
    res[:, :, 2] = b
    # res = np.array([r,g,b])
    cv2.imshow('hehe', res)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()





