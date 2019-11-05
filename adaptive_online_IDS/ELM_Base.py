import numpy as np
from sklearn import metrics


class ELM_BASE(object):

    def __init__(self,input_feature,num_hidden,output_feature):
        self.num_feature = input_feature
        self.num_hidden = num_hidden
        self.output_feature = output_feature
        self.GenerateModel()

    def GenerateModel(self):
        self.w = np.random.uniform(-1, 1, (self.num_feature,self.num_hidden))#W:2*100
        bias = np.random.uniform(-1, 1, (1, self.num_hidden))
        self.beta = np.zeros([self.num_hidden+self.num_feature,self.output_feature],dtype=np.float)
        self.first_b = bias
        self.H_inv_train=None
        self.H_train=None
        self.trained=False
    
    def AddNewType(self):
        self.num_feature=self.num_feature+1
        new_w=np.random.uniform(-1,1,(1,self.num_hidden))
        self.w=np.row_stack((self.w,new_w))
        new_beta=np.zeros([1,self.output_feature],dtype=float)
        self.beta=np.row_stack((self.beta,new_beta))
        self.trained=False
        self.UpperLevelAdd()

    def UpperLevelAdd(self):
        pass
    
    def reconstruct_BaseModel(self,input_feature,num_hidden,output_feature):
        new_added_feature=input_feature-self.num_feature
        self.num_feature = input_feature
        self.num_hidden = num_hidden
        self.output_feature = output_feature
        new_w= np.random.uniform(-1, 1, (new_added_feature, self.num_hidden))
        self.w=np.concatenate((self.w,new_w),axis=0)


    def sigmoid(self, x):
        #return np.where(x<0,0,x)
        return 1.0 / (1 + np.exp(-x))

    def rule(self,x):
        return np.where(x<0,0,x)

    def getH_InvTrain(self):
        return self.H_inv_train

    def getBiasArray(self,num_data):
        b = self.first_b
        for _ in range(num_data - 1):
            b = np.row_stack((b, self.first_b))
        return b

    def comput_H_train(self,x_train,H_need_append,H_append_matrix):
        num_data = len(x_train)
        b = self.getBiasArray(num_data)
        mul = np.matmul(x_train,self.w)
        add = mul + b
        H = self.sigmoid(add)
        if not H_need_append:
            H_train = H
        else:
            H_train = np.concatenate((H, H_append_matrix), axis=1)
        return H_train


    def train(self, x_train, y_train,loss_function,H_need_append=False,H_append_matrix=None,retrain=False):
        if not self.trained or retrain:
           H_train=self.comput_H_train(x_train,H_need_append,H_append_matrix)#x_train:100*2,H_need_append:True,H_append_matrix
           self.H_train=H_train
           H_inv = np.linalg.pinv(self.H_train)
           self.H_inv_train = H_inv
           self.trained=True
        else:
            H_inv = self.H_inv_train
        self.beta = np.matmul(H_inv, y_train)
        return loss_function(x_train,y_train)

    def predict(self, x_test,y_test,cost_function):
        H=self.comput_H_train(x_test,True,x_test)
        pred_Y=np.matmul(H,self.beta)
        #pred_Y = np.dot(self.sigmoid(np.dot(t_data, self.w) + b), self.beta)
        score=cost_function(pred_Y,y_test)
        '''
        self.predy = []
        for i in pred_Y:
            L = i.tolist()
            self.predy.append(L.index(max(L)))
        '''
        return score