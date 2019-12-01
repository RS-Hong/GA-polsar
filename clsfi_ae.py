import numpy as np
#19.10.25

def clsfi_ae(check_data,result):
    '''利用验证样本数据对分类结果验证
    check_data 输入的验证样本数据，
    result 表示分类结果
    lines 原始数据行数 samples 原始数据列数'''
    #check_num 类别数
    lines,samples = result.shape
    check_num = len(check_data)
    cls_data = np.zeros((lines,samples),dtype = 'float')
    for i in range(check_num):
        #1.以i来记录类别编号，
        #2.以循环对每个类别的分布编号
        sam_num = int(len(check_data[i])/4) #每类别样本数
        samp = check_data[i]
        for j in range(sam_num):
            x1,y1,x2,y2 = samp[4*j],samp[4*j+1],samp[4*j+2],samp[4*j+3]
            cls_data[x1:x2,y1:y2] = int(i) + 1
    cls_data[np.where(check_data == 0)] = np.nan
    #得到验证类别结果后分别计算各类别与分类结果之间的精度
    #1.构造混淆矩阵
    #2.对第一个类别掩模
    #3.计算正确分类的个数和其他类别分类为第一个类别的个数
    #4.构成第一行
    #5.计算其余精度评价参数
    #6.循环
    cfx_mx = np.zeros((check_num, check_num), dtype=int)  # 初始化混淆矩阵
    num = 0
    total_num = 0
    for i in range(check_num):
        for j in range(check_num):
            cfx_mx[i,j] = len(np.where(result[np.where(cls_data == i + 1)] == j + 1)[0])
            total_num += cfx_mx[i, j]
            if i == j:
                num += cfx_mx[i,j]
        #     print("%6d" % (len(np.where(result[np.where(cls_data == i + 1)] == j + 1)[0])),end='')
        # print("")
    return  num/total_num,cfx_mx
