import re
import math
import random
import readbin
import numpy as np
import ga_svm
import os
from  multiprocessing import Process, Lock, Manager

# 1.选出初始种群
# 2.计算适应度函数 
# 3.交叉变异产生新的种群 
# 4.最优适应度值不再变化
RETAIN_RATE = 0.2
RANDOM_SELECT_RATE = 0.5
VARIATION_RATE = 0.05

class GATS(object):
    """docstring for GATS"""
    def __init__(self, dim,count,dataset,mdict,mlist):
        super(GATS, self).__init__()
        """
        self.dim 表示初始维度即染色体长度
        self.ininum初始种群大小
        """
        self.dim = dim
        self.ininum = count
        self.dataset = dataset
        self.mdict = mdict
        self.mlist = mlist
        self.lock = Lock()
        self.population = self.gen_population(dim,count)    #每次迭代的种群
        self.fit_population = []    #将计算过适应度函数的染色体和适应度值存入列表中避免重复计算
        self.fit_value = []


    def gen_population(self,dim,count):
        '''生成初始种群'''
        population = []
        for i in range(count):
            chromosome = self.gen_chromosome(dim)
            while chromosome in population:
                #判断染色体是否已经存在
                chromosome = self.gen_chromosome(dim)                
            population.append(chromosome)
        return population
    
    def gen_chromosome(self,dim):
        #随机产生一个染色体10进制test_chromosome
        #转化成二进制表示
        #转化生列表形式
        tem_chromosome = random.randint(0,2**dim-1)
        bchromosome = bin(tem_chromosome)
        res = re.match(r"0b(.+)",bchromosome)
        tem_chromosome = res.group(1)
        num = dim-len(tem_chromosome)
        tem_chromosome = "0" * num + tem_chromosome
        tem_chromosome = [i for i in tem_chromosome]
        return tem_chromosome

    def evolve(self):
        retain_tup = self.retain()
        self.crossover(retain_tup,2)
        self.variation()

    def fitness(self,chromosome, lock, mdict, mlist):
        '''
        # self.best等变量，由于多进程的存在，每个进程都会带有一分完整的资源，使得内部的修改对其他进程不会生效。
        # 19.12.3修改为通过文件存储来保存多进程的修改
        # 19.12.4增加进程锁，保护文件内容安全
        # 19.12.8修改为利用manager共享变量判断最优解

        1.计算每条染色体的适应度值
        2.将最大的适应度及对应的染色体保存进self.best,self.final_res用来记录最终的最优结果

        dataset : 根据染色体形状得到对应特征空间数据集
        num: 分类准确度
        cfx_mx: 混淆矩阵
        result: 分类结果
        fit_value: 适应度函数值
        '''
        currentpath = os.getcwd()
        clsresultpath = os.path.join(currentpath, 'cls_result.txt')
        cfxmxpath = os.path.join(currentpath, 'cfx_mx.txt')
        
        npch = np.array(list(map(int,chromosome)))  #将chromosome转化为int型的array
        dataset = self.dataset[:,:,np.where(npch == 1)]
        x,y,m,k = dataset.shape
        dataset = dataset.reshape(x,y,k)
        num, cfx_mx, result = ga_svm.msvm(dataset)   
        fit_value = 0.8 * (len(chromosome)-sum(list(map(int,chromosome))))/len(chromosome) + num   #适应度函数公式
        lock.acquire()
        temp = mdict
        mlist.append((chromosome,fit_value))
        print(mlist)
        if mdict['best'] == 0:
            #如best不存在，则创建新的文件用以存放分类结果result，混淆矩阵cfx_mx
            temp['best'] = [chromosome, fit_value]
            np.savetxt(clsresultpath, np.array(result), fmt='%d', delimiter=',')
            np.savetxt(cfxmxpath, np.array(cfx_mx), fmt='%d', delimiter=',')
            mdict = temp
            print(mdict)
        else:
            #如果存在则取出best去最新的best比较，将最优的best覆盖原有文件
            if fit_value > temp['best'][1]:
                temp['best'] = [chromosome,fit_value]
                np.savetxt(clsresultpath, np.array(result), fmt='%d', delimiter=',')
                np.savetxt(cfxmxpath, np.array(cfx_mx), fmt='%d', delimiter=',')
                mdict = temp
                print(mdict)
        lock.release()
        return fit_value


    def retain(self):
        '''
        19.12.8修改利用manager模块共享变量判断最优解
        得到种群保留下来的染色体，从大到小排列
        比例为RATAIN_RATE个染色体是直接保留的，
        比例为RANDOM_SELECT_RATE个是随机保留的
        '''
        fit_list = []   #population中所有的chromosome及对应的fitvalue
        chro_list = []
        new_fitlist = []    #未计算过的chromosome及对应的fitvalue
        for chromosome in self.population:
            if chromosome in self.fit_population:
                fit_list.append((chromosome,self.fit_value[self.fit_population.index(chromosome)]))
            else:
                #加进程？
                # chromosome_fitness = self.fitness(chromosome)
                # fit_list.append((chromosome,chromosome_fitness))
                # self.fit_population.append(chromosome)
                # self.fit_value.append(chromosome_fitness)
                #2019.11.13修改增加进程
                chro_list.append(chromosome)
        if len(chro_list) != 0:
            if len(chro_list) % 4 == 0:
                for i in range(int(len(chro_list) / 4)):
                    p1 = Process(target = self.fitness, args = (chro_list[i * 4], self.lock, self.mdict, self.mlist))
                    p2 = Process(target = self.fitness, args=(chro_list[i * 4 + 1], self.lock, self.mdict, self.mlist))
                    p3 = Process(target = self.fitness, args=(chro_list[i * 4 + 2], self.lock, self.mdict, self.mlist))
                    p4 = Process(target = self.fitness, args=(chro_list[i * 4 + 3], self.lock, self.mdict, self.mlist))
                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()
                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
            elif len(chro_list) % 4 == 1 :
                for i in range(int(len(chro_list)/4)):
                    p1 = Process(target=self.fitness,args=(chro_list[i * 4],self.lock, self.mdict, self.mlist))
                    p2 = Process(target=self.fitness, args=(chro_list[i * 1], self.lock, self.mdict, self.mlist))
                    p3 = Process(target=self.fitness, args=(chro_list[i * 2], self.lock, self.mdict, self.mlist))
                    p4 = Process(target=self.fitness, args=(chro_list[i * 3], self.lock, self.mdict, self.mlist))
                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()
                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
                p1 = Process(target=self.fitness, args=(chro_list[-1], self.lock, self.mdict, self.mlist))
                p1.start()
                p1.join()
            elif len(chro_list) % 4 == 2:
                for i in range(int(len(chro_list)/4)):
                    p1 = Process(target=self.fitness, args=(chro_list[i * 4], self.lock, self.mdict, self.mlist))
                    p2 = Process(target=self.fitness, args=(chro_list[i * 1], self.lock, self.mdict, self.mlist))
                    p3 = Process(target=self.fitness, args=(chro_list[i * 2], self.lock, self.mdict, self.mlist))
                    p4 = Process(target=self.fitness, args=(chro_list[i * 3], self.lock, self.mdict, self.mlist))
                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()
                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
                p1 = Process(target=self.fitness, args=(chro_list[-2], self.lock, self.mdict, self.mlist))
                p2 = Process(target=self.fitness, args=(chro_list[-1], self.lock, self.mdict, self.mlist))
                p1.start()
                p2.start()
                p1.join()
                p2.join()
            elif len(chro_list) % 4 == 3:
                for i in range(int(len(chro_list)/4)):
                    p1 = Process(target=self.fitness, args=(chro_list[i * 4], self.lock, self.mdict, self.mlist))
                    p2 = Process(target=self.fitness, args=(chro_list[i * 1], self.lock, self.mdict, self.mlist))
                    p3 = Process(target=self.fitness, args=(chro_list[i * 2], self.lock, self.mdict, self.mlist))
                    p4 = Process(target=self.fitness, args=(chro_list[i * 3], self.lock, self.mdict, self.mlist))
                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()
                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
                p1 = Process(target=self.fitness, args=(chro_list[-3], self.lock, self.mdict, self.mlist))
                p2 = Process(target=self.fitness, args=(chro_list[-2], self.lock, self.mdict, self.mlist))
                p3 = Process(target=self.fitness, args=(chro_list[-1], self.lock, self.mdict, self.mlist))
                p1.start()
                p2.start()
                p3.start()
                p1.join()
                p2.join()
                p3.join()
            new_fitlist = self.mlist[:]
            fit_list = fit_list + new_fitlist
            for l in new_fitlist:
                #将新得到的fitlist列表加入库中
                self.fit_population.append(l[0])
                self.fit_value.append(l[1])
        fit_list.sort(key = lambda x:x[1],reverse = True)
        retain_num = math.ceil(self.ininum * RETAIN_RATE)
        retain_tup = fit_list[:retain_num]
        for chromosome in fit_list[retain_num:]:
            if random.random() < RANDOM_SELECT_RATE:
                retain_tup.append(chromosome)
        self.mlist[:] = []
        return retain_tup

    def crossover(self,retain_tup,method):
        '''交叉算子'''
        children = []
        chil_len = self.ininum - len(retain_tup)
        if method == 1:                
            # 1.单点交叉
            while len(children) < chil_len:
                male = random.randint(0,len(retain_tup)-1)
                female = random.randint(0,len(retain_tup)-1)
                if male != female:
                   #随机选择交叉点(单点交叉，如果剩余容量大于1则交叉后两条染色体都加入种群，否则只加入一条)
                    cross_pos = random.randint(0, self.dim-1)
                    if chil_len - len(children) > 1:
                        male = retain_tup[male][0]
                        female = retain_tup[female][0]
                        chil_male = male[:cross_pos] + female[cross_pos:]
                        chil_female = female[:cross_pos] + male[cross_pos:]
                        children.append(chil_male)
                        children.append(chil_female)
                    else:
                        male = retain_tup[male][0]
                        female = retain_tup[female][0]
                        chil_male = male[:cross_pos] + female[cross_pos:]
                        children.append(chil_male)        
            self.population = [i[0] for i in retain_tup] + children
        elif method == 2:
            #两点交叉
            while len(children) < chil_len:
                male = random.randint(0,len(retain_tup)-1)
                female = random.randint(0,len(retain_tup)-1)
                if male != female:
                    cross_pos1 = random.randint(0,self.dim-1)
                    if cross_pos1 < self.dim:
                        cross_pos2 = random.randint(cross_pos1 + 1,self.dim)
                        if chil_len - len(children) > 1:
                            male = retain_tup[male][0]
                            female = retain_tup[female][0]
                            chil_male = male[:cross_pos1] + female[cross_pos1:cross_pos2] + male[cross_pos2:]
                            children.append(chil_male)
                            chil_female = female[:cross_pos1] + male[cross_pos1:cross_pos2] + female[cross_pos2:]
                            children.append(chil_male)
                            children.append(chil_female)
                        else:
                            male = retain_tup[male][0]
                            female = retain_tup[female][0]
                            chil_male = male[:cross_pos1] + female[cross_pos1:cross_pos2] + male[cross_pos2:]
                            children.append(chil_male)
            self.population = [i[0] for i in retain_tup] + children


    def variation(self):
        '''变异算子'''
        for i in range(len(self.population)):
            if random.random() < VARIATION_RATE:
                #变异位置
                #要变异的染色体
                pos = random.randint(0,len(self.population[0]) - 1)
                var_chrom = self.population[i]
                fit_value = self.fit_value[self.population.index(var_chrom)]
                if var_chrom[pos] == '0':
                    var_chrom[pos] = '1'
                else:
                    var_chrom[pos] = '0'
                if var_chrom not in self.fit_population:
                    var_fitvale = self.fitness(var_chrom, self.lock, self.mdict, self.mlist)
                    if var_fitvale > fit_value:
                        # 防止变异摧毁好基因
                        self.population[i] = var_chrom
                    self.fit_population.append(var_chrom)
                    self.fit_value.append(var_fitvale)



def main():
    manager = Manager()
    mdict = manager.dict()
    mlist = manager.list()
    mdict['best'] = 0
    fit_pool = []  # 适应度函数值池,用以绘制折线图
    directory = "C:\\Users\\Administrator\\Desktop\\AIRSAR_Flevoland_LEE\\T3"
    dataset, trait_pool = readbin.readallbin(directory, 750, 1024)
    x, y, k = dataset.shape
    gats = GATS(k,24,dataset,mdict,mlist)
    j = 0
    #1.j控制进化代数
    #2.用for循环设定初始进化代数，并将最优染色体对应的结果与混淆矩阵绘制
    i_t = 100 #控制进化代数
    for i in range(i_t):
        gats.evolve()
        j += 1
        print('第 %d 次迭代正在进行' % (j))
        fit_pool.append(mdict['best'][1])
        if j == i_t:
            readbin.draw_fit(fit_pool)
            final_res = np.loadtxt("C:\\Users\\Administrator\\Desktop\\原始\\github\\GA\\cls_result.txt", delimiter=',')
            readbin.draw_cls(final_res,15)
            cfx_mx = np.loadtxt(os.path.join(os.getcwd(), 'cfx_mx.txt'), delimiter=',')
            x,y = cfx_mx.shape
            for i in range(x):
                for j in range(y):
                    print("%6d" % (cfx_mx[i,j]),end='')
                print("")
        print(mdict['best'])

if __name__ == '__main__':
    main()
