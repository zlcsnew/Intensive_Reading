#!-*- coding:utf-8 -*-
import numpy as np


class UpdateManager:
    def __init__(self,cluster):
        self.cluster=cluster
        H=self.cluster.H_train
        self.P=np.linalg.pinv(np.matmul(H.T,H))
    
    def getOneHotCode(self,data_len,label):
        clustering_number=self.cluster.clusting_type
        HotCode=np.zeros([data_len,clustering_number],dtype=int)
        for i in range(data_len):
            HotCode[i,label[i]]=1
        return HotCode

    def UM_Procedure_1(self, newdata_len, data, label=None):
        if label is None:
            print "\n\n\ngenerate labels\n\n\n"
            label=np.random.randint(0,self.cluster.clusting_type,newdata_len)
        x_train=self.getOneHotCode(newdata_len,label)
        print x_train
        H_tmp=self.cluster.comput_H_train(x_train,True,x_train)
        #H_new=np.concatenate([self.cluster.H_train,H_tmp],axis=0)\
        H_new=H_tmp
        I_hnew_pold_hnew=np.eye(np.size(H_new,axis=0))+np.matmul(H_new,np.matmul(self.P,H_new.T))
        I_hnew_pold_hnew_inv=np.linalg.inv(I_hnew_pold_hnew)
        self.P=self.P-np.matmul(self.P,np.matmul(H_new.T,np.matmul(I_hnew_pold_hnew_inv,np.matmul(H_new,self.P))))
        X_Hnew_beta=data-np.matmul(H_new,self.cluster.beta)
        self.cluster.beta=self.cluster.beta+np.matmul(self.P,np.matmul(H_new.T,X_Hnew_beta))
        self.cluster.H_train=H_new
        return self.cluster

    def UM_Procedure_2(self,data_len,data,label):
        '''
           方法:
               首先，已经通过了Decision_Maker的evaluation_cluster求得了这个X最适合的聚类；
               其次，通过这个最适合的label和对应的data，计算beta
        '''
        x_train=self.getOneHotCode(data_len,label)
        H_tmp=self.cluster.comput_H_train(x_train,True,x_train)
        self.cluster.beta=np.matmul(np.linalg.pinv(H_tmp),data)
        self.cluster.H_train=H_tmp
        return self.cluster

    def UM_Procedure_3(self,data_len,data,label):
        self.cluster.AddNewType()
        one_hot_code=self.getOneHotCode(data_len,label)
        self.cluster.train(one_hot_code, data, self.cluster.get_result, True, one_hot_code,True)
        return self.cluster