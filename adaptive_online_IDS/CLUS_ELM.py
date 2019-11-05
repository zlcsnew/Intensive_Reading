import matplotlib
import ELM_Base
import numpy as np
import random
import matplotlib.pyplot as plt

class ELM_ERROR(Exception):
    def __init__(self,reason):
        self.args = reason

class CLUS(ELM_Base.ELM_BASE):
    def __init__(self,input_feature,num_hidden,output_feature,iter_times,clusting_rate):
        super(CLUS,self).__init__(input_feature,num_hidden,output_feature)
        self.clusting_type = len(clusting_rate)#number of clusting
        if self.clusting_type!=input_feature:#input_feature == number of clusting
            raise ELM_ERROR("clusting number doesn't fit input feature")
        self.iter_times=iter_times
        self.clusting_rate=clusting_rate
        self.X_train=None
        self.defalut_loss_value=0.10
        self.Supervision=True
        self.color=["red","green","black","cyan","magenta"]
        self.type=["x","*","^","s","D"]
        self.now_data_number=0
    
    def UpperLevelAdd(self):
        self.clusting_type=self.clusting_type+1

    def getOneHotCode(self,shape,clusting_number):#run many times
        code = np.zeros((shape[0],shape[1]), dtype=int)#data_number,type_number
        now_start_index=0
        for index in range(shape[1]):
            now_end_index=int(now_start_index+clusting_number[index])
            now_start_index=int(now_start_index)
            tmp=np.zeros(shape[1],dtype=int)
            tmp[index]=1
            code[now_start_index:now_end_index]=tmp
            now_start_index=now_end_index
        return code

    def loss_function(self,labels,x_data):
        number_of_data=len(labels)
        mul = np.dot(self.H_train,self.beta)
        loss_matrix = np.abs(mul-x_data)
        loss=np.zeros(number_of_data,dtype=float)
        for index in range(self.output_feature):
            loss=loss+np.square(loss_matrix[:,index])
        loss=loss/self.output_feature
        loss_value=sum(loss)/number_of_data
        return loss_value,loss

    def Swap(self,loss,origin_input_data,clusting_number,clusting_result,swap_rate=0.3):
        swap_candidate_number=0
        tmp_loss=-1*loss
        sorted_loss_index=np.array([],dtype=int)
        start_index=0
        for now_clusting_number in clusting_number:
            now_sorted_index_total=tmp_loss[start_index:start_index+now_clusting_number].argsort()
            now_sorted_index_total=start_index+now_sorted_index_total
            now_sorted_index=now_sorted_index_total[0:int(now_clusting_number*swap_rate)]
            swap_candidate_number=swap_candidate_number+int(now_clusting_number*swap_rate)
            sorted_loss_index=np.concatenate([sorted_loss_index,now_sorted_index],axis=0)
            start_index=start_index+now_clusting_number
        #print sorted_loss_index,'\n'
        #swap_candidate_index=sorted_loss_index[0:swap_candidate_number]
        swap_candidate_index=sorted_loss_index
        i=0
        while i<swap_candidate_number:
            i_swap_index=swap_candidate_index[i]
            swap_cluster=clusting_result[i_swap_index,:]
            j=i+1
            while j<swap_candidate_number:
                j_swap_index=swap_candidate_index[j]
                if (swap_cluster==clusting_result[j_swap_index,:]).all():
                    j=j+1
                    continue
                tmp_data=origin_input_data[i_swap_index,:]
                origin_input_data[i_swap_index,:]=origin_input_data[j_swap_index,:]
                origin_input_data[j_swap_index,:]=tmp_data
                swap_candidate_index=np.delete(swap_candidate_index,j)
                swap_candidate_number=swap_candidate_number-1
                break
            i=i+1
        return origin_input_data

    def fit(self,input_origin_data,need_plot,restart=False):
        number_of_input_data = np.size(input_origin_data,0)
        self.now_data_number = number_of_input_data
        print "number of data and cluster:",self.now_data_number,self.clusting_rate
        clusting_number=range(self.clusting_type)
        for index in range(self.clusting_type):
            clusting_number[index]=int(number_of_input_data*self.clusting_rate[index])#checked
        shape=[number_of_input_data,self.clusting_type]#checked
        clusting_result=self.getOneHotCode(shape,clusting_number)#checked
        #print "training input:",clusting_result,input_origin_data
        loss_array = []
        #x_train, y_train,loss_function,H_need_append=False,H_append_matrix=None
        loss_value, loss = self.train(clusting_result, input_origin_data, self.loss_function,H_need_append=True,H_append_matrix=clusting_result,retrain=restart)
        loss_array.append(loss_value)
        print("inital:"+str(loss_value))
        turn=1
        while turn<=self.iter_times:
            old_data=input_origin_data
            old_loss_value,old_loss = loss_value,loss
            #print old_data
            input_origin_data=self.Swap(loss,input_origin_data,clusting_number,clusting_result)
            loss_value, loss = self.train(clusting_result, input_origin_data, self.loss_function)
            if loss_value<=old_loss_value:
                loss_array.append(loss_value)
                print("No:" + str(turn) + "   " + str(loss_value))
            else:
                input_origin_data=old_data
                loss_value,loss = old_loss_value,old_loss
            turn+=1

        if need_plot:
            x=range(len(loss_array))
            plt.plot(x, loss_array)
            plt.show()


    def get_centor_point(self):
        clusting_number=np.ones(self.clusting_type,dtype=int)
        shape=[self.clusting_type,self.clusting_type]
        code=self.getOneHotCode(shape,clusting_number)
        centor=self.predict(code,None,self.get_result)
        return centor

    def get_result(self,predit,label):
        return predit


    def UM_Procedure_1(self,newdata,label=None):
        newdata_len=len(newdata)
        #random_assign=False
        if label is None:
            #random_assign=True
            label=np.random.randint(0,self.clusting_type,newdata_len)
        self.now_data_number+=newdata_len
        tmp_newdata=np.zeros([newdata_len,self.output_feature],dtype=float)
        tmp_newdata_number=np.zeros(self.clusting_type,dtype=int)
        now_start=0
        centor=[]
        for index in range(self.clusting_type):
            tmp_index=np.where(label==index)[0]
            tmp_number=len(tmp_index)
            #print tmp_number
            tmp_newdata[now_start:now_start+tmp_number]=newdata[tmp_index,:]
            tmp_newdata_number[index]=tmp_number
            tmp_centor=np.mean(tmp_newdata[tmp_index,:],axis=0)
            centor.append(tmp_centor)
            now_start=now_start+tmp_number
        print("new add centor:",centor)
        shape=[len(newdata),self.clusting_type]
        OneHotCode=self.getOneHotCode(shape,tmp_newdata_number)
        H_chunk=self.comput_H_train(OneHotCode,True,OneHotCode)
        H_new=np.concatenate([self.H_train,H_chunk],axis=0)
        P_old_inv=np.dot(self.H_train.T,self.H_train)
        p_old=np.linalg.pinv(P_old_inv)
        tmp_result=np.linalg.pinv(np.eye(len(H_new))+H_new.dot(p_old.dot(H_new.T)))
        P_new=p_old-p_old.dot(H_new.T.dot(tmp_result.dot(H_new.dot(p_old))))
        print np.shape(self.beta),np.shape(P_new),np.shape(H_new),np.shape(tmp_newdata)
        beta_new=self.beta+P_new.dot(H_chunk.T.dot(tmp_newdata+H_chunk.dot(self.beta)))
        self.beta=beta_new

    def UM_Procedure_2(self,x_train,clusting_number):
        self.clusting_rate=(clusting_number.astype(dtype=float))/np.sum(clusting_number)
        self.fit(x_train,True,True)

    def UM_Procedure_3(self,x_train_old,cluster_number_old,x_train_new,cluster_number_new):
        x_train=np.concatenate((x_train_old,x_train_new),axis=0)
        #print(cluster_number_old,cluster_number_new)
        clusting_number=np.concatenate((cluster_number_old,cluster_number_new),axis=0)
        #clusting_number=np.append(cluster_number_old,cluster_number_new)
        self.clusting_rate=(clusting_number.astype(dtype=float))/np.sum(clusting_number)
        self.clusting_type=len(clusting_number)
        self.reconstruct_BaseModel(self.clusting_type,self.num_hidden,self.output_feature)
        self.fit(x_train,True,True)
