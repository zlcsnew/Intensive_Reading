import CLUS_ELM
import numpy as np
import matplotlib.pyplot as plt

class DecisionMaker:
    def __init__(self,cluster):
        self.cluster=cluster
        self.Decision={1: "assignment is not suitable", 2 : "need add new cluster" ,3:"No problem"}
        self.Thershold=0.05

    def evalate_enhanced_clusting(self,clusting_enhanced,x_train):
        centor=[]
        new_data = np.zeros([len(x_train),self.cluster.output_feature],dtype=float)
        new_data_number = np.zeros(self.cluster.clusting_type,dtype=int)
        now_start_index=0
        for index in range(self.cluster.clusting_type):
            tmp_clusting=np.where(clusting_enhanced==index)[0]#finding index-th cluster
            tmp_sample=x_train[tmp_clusting,:]#finding index-th data
            now_end_index=now_start_index+len(tmp_sample)#insert into new_data
            new_data_number[index]=len(tmp_sample)#the index-th cluster has len(sample) data
            new_data[now_start_index:now_end_index]=tmp_sample#insert into new_data
            tmp_centor=np.mean(new_data[now_start_index:now_end_index],axis=0)
            now_start_index = now_end_index
            centor.append(tmp_centor)
        return centor,new_data,new_data_number

    def getLabel(self,loss_array,x_train):
        inv_M=1 / np.sum(loss_array, axis=1, keepdims=True)
        score=loss_array*inv_M#loss array  uniform
        max_index=np.argsort(score)[:,0:2]
        return max_index[:,0]


    def evaluation_cluster(self,x_train):
        centor=self.cluster.get_centor_point()#[[1,0,0],[0,1,0],[0,0,1]]
        sample_number=np.size(x_train,axis=0)
        loss_array=np.zeros([sample_number,self.cluster.clusting_type],dtype=float)
        for clus_index in range(self.cluster.clusting_type):
            centor_array=np.tile(centor[clus_index,:],(sample_number,1))
            tmp_ESM_array=np.square(np.abs(centor_array-x_train))
            ESM_value=np.sum(tmp_ESM_array,axis=1,keepdims=True)/self.cluster.output_feature
            loss_array[:,clus_index]=ESM_value[:,0]
            #print "\n\n\ntmp_ESM_array and ESM_value and loss_array"
            #print tmp_ESM_array
            #print ESM_value
            #print loss_array
        clustering_type = self.getLabel(loss_array,x_train)
        return clustering_type

    def need_add_cluster(self,x_train):
        centor=self.cluster.get_centor_point()#[[1,0,0],[0,1,0],[0,0,1]]
        sample_number=np.size(x_train,axis=0)
        loss_array=np.zeros([sample_number,self.cluster.clusting_type],dtype=float)
        for clus_index in range(self.cluster.clusting_type):
            centor_array=np.tile(centor[clus_index,:],(sample_number,1))
            tmp_ESM_array=np.square(np.abs(centor_array-x_train))
            ESM_value=np.sum(tmp_ESM_array,axis=1,keepdims=True)/self.cluster.output_feature
            loss_array[:,clus_index]=ESM_value[:,0]
        need_add,clustering_type = self.CheckCluster(loss_array,x_train)
        return need_add,clustering_type

    def CheckCluster(self,loss_array,x_train):
        needed=False
        inv_M=1 / np.sum(loss_array, axis=1, keepdims=True)
        score=loss_array*inv_M#loss array  uniform
        max_index=np.argsort(score)[:,0:2]
        second_cluster=max_index[:,1]
        first_cluster=max_index[:,0]
        data_number=np.size(x_train,0)
        for i in range(data_number):
            distace_diff=score[i,second_cluster[i]]-score[i,first_cluster[i]]
            if distace_diff<self.Thershold:
                max_index[i,0]=self.cluster.clusting_type
                needed=True
        return needed,max_index[:,0]