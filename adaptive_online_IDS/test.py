##! -*- coding:utf-8 -*-
import CLUS_ELM
import Decision_Maker
import Update_Manager
import numpy as np
import matplotlib.pyplot as plt
def gengerate_example(number,need_shiffle=True):
    example=np.zeros((sum(number),2),dtype=float)#example:N*2
    now_start=0
    for i in range(len(number)):
        if i==2:
            mean1 = [(i+1)*10, (i+1)*10]
        elif i==3:
            mean1 = [(i-1)*10, (i-1)*10]
        else:
            mean1 = [(i)*10, (i)*10]
        cov1 = [[1, 0], [0, 1]]
        data = np.random.multivariate_normal(mean1, cov1, number[i])
        example[now_start:now_start+number[i]]=data
        now_start+=number[i]
        if need_shiffle:
            np.random.shuffle(example)
    #plt.scatter(example[:,0],example[:,1])
    #plt.show()
    return example

if __name__=="__main__":
    classfier = CLUS_ELM.CLUS(3,100,2,100,[0.5,0.2,0.3])##注意：input_future_number是标签数量，输出神经元个数是向量的维度
    example=gengerate_example([20,30,50])
    need_continue=True
    classfier.fit(example,False)
    centor=classfier.get_centor_point()
    print "old centor",centor
    #初始化训练结束

    DM=Decision_Maker.DecisionMaker(classfier)
    updatamanager=Update_Manager.UpdateManager(classfier)
    better_clustering_result=DM.evaluation_cluster(example)
    #print "after evaluation" 
    #print better_clustering_result  
    color=['r','b','y']
    for i in [0,1,2]:
        index=np.where(better_clustering_result==i)[0]
        plt.plot(example[index,0],example[index,1],color[i]+"o")
    plt.savefig("evaluation_result.png")
    plt.cla()   
    #plt.show()
    classfier=updatamanager.UM_Procedure_2(100,example,better_clustering_result)
    centor=classfier.get_centor_point()
    print "better centor",centor
    DM=Decision_Maker.DecisionMaker(classfier)
    test_example=gengerate_example([50,50,50])
    print test_example
    result=DM.evaluation_cluster(test_example)
    color=['r','b','y']
    for i in [0,1,2]:
        index=np.where(result==i)[0]
        plt.plot(test_example[index,0],test_example[index,1],color[i]+"o")
    plt.savefig("after_evaluation_updating_result.png")
    plt.cla()
    #经过Desion_maker 和 updatemanager 求得优化后的clustering
    
    DM=Decision_Maker.DecisionMaker(classfier)
    test_example=gengerate_example([50,50,50])
    clustering_result=DM.evaluation_cluster(test_example)
    color=['r','b','y']
    print "test data!"
    for i in [0,1,2]:
        index=np.where(clustering_result==i)[0]
        plt.plot(test_example[index,0],test_example[index,1],color[i]+"o")
    plt.savefig("test_data_result.png")
    plt.cla()   
    #plt.show()
    #添加测试数据
    
    retrain_data_number=100
    retrain_cluster_number=[int(0.5*retrain_data_number),int(0.2*retrain_data_number),int(0.3*retrain_data_number)]
    labels=[0]*retrain_data_number#np.zeros([1,retrain_data_number],dtype=int)
    example=gengerate_example(retrain_cluster_number,False)
    centor=classfier.get_centor_point()
    labeled_index=np.argsort(centor[:,0])
    
    now_start=0
    for i in range(3):
        labels[now_start:now_start+retrain_cluster_number[i]]=[labeled_index[i]]*retrain_cluster_number[i]
        now_start=now_start+retrain_cluster_number[i]
    updatamanager=Update_Manager.UpdateManager(classfier)
    classfier=updatamanager.UM_Procedure_1(retrain_data_number,example,labels)
    centor=classfier.get_centor_point()
    print "new centor",centor

    DM=Decision_Maker.DecisionMaker(classfier)
    test_example=gengerate_example([50,50,50])
    clustering_result=DM.evaluation_cluster(test_example)
    color=['r','b','y']
    print "after Updating and test the data"
    #print clustering_result
    for i in [0,1,2]:
        index=np.where(clustering_result==i)[0]
        plt.plot(test_example[index,0],test_example[index,1],color[i]+"o")
    plt.savefig("after_updating.png")
    plt.cla()   
    #plt.show()
    #添加新的数据重新训练并测试
    
    
    retrain_data_number=100
    retrain_cluster_number=[int(0.25*retrain_data_number),int(0.25*retrain_data_number),int(0.25*retrain_data_number),int(0.25*retrain_data_number)]
    labels=[0]*retrain_data_number#np.zeros([1,retrain_data_number],dtype=int)
    example=gengerate_example(retrain_cluster_number,False)
    print example
    print "want to add a new cluster"
    DM=Decision_Maker.DecisionMaker(classfier)
    needed,clustering_result=DM.need_add_cluster(example)
    print clustering_result
    if needed:
        print "begin to add a new cluster"
        UM=Update_Manager.UpdateManager(classfier)
        classfier=UM.UM_Procedure_3(retrain_data_number,example,clustering_result)
        centor=classfier.get_centor_point()
        print "new centor",centor
    DM=Decision_Maker.DecisionMaker(classfier)
    clustering_result=DM.evaluation_cluster(example)
    color=['r','b','y','g']
    for i in [0,1,2,3]:
        index=np.where(clustering_result==i)[0]
        plt.plot(example[index,0],example[index,1],color[i]+"o")
    plt.savefig("after_add_evaluate_cluster.png")
    plt.cla()   
    #plt.show()