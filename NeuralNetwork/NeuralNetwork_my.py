from sklearn.datasets import load_iris
import numpy as np
from IPython import embed
# from sklearn import datasets
iris = load_iris()


def sigmod(z):
    h = 1./(1+np.exp(-z))
    return h
    
def de_sigmoid(z,h):
    return h*(1-h)
    
def relu(z):
    h = np.maximum(z, 0)
    return h
    
def de_relu(z,h):
    z[z <= 0] = 0
    z[z > 0] = 1.0
    return z

def no_active(z):
    h = z
    return h

def de_no_active(z,h):
    return np.ones(h.shape)
  
def loss_L2(o,lab):
    diff = lab-o
    sqrDiff = diff ** 2
    return 0.5*np.sum(sqrDiff)
    
def de_loss_L2(o,lab):
    return o-lab


def loss_CE(o,lab):    
    p = np.exp(o)/np.sum(np.exp(o),axis=1,keepdims=True)
    loss_ce = np.sum(-lab*np.log(p))
    return loss_ce

def de_loss_CE(o,lab):
    p = np.exp(o)/np.sum(np.exp(o),axis=1,keepdims=True)
    return p-lab

def bulid_net(dim_in,list_num_hidden,
              list_act_funs,list_de_act_funs):
    layers=[]
    
    # 逐层的进行网络构建
    for i in range(len(list_num_hidden)):
        layer = {}
        
        # 定义每一层的权重
        if i ==0:
            # layer["w"]= 0.2*np.random.randn(dim_in,list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"]= 0.01*np.random.randn(dim_in,list_num_hidden[i])  # 用relu 激活函数
        else:
            # layer["w"]= 0.2*np.random.randn(list_num_hidden[i-1],list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"]= 0.01*np.random.randn(list_num_hidden[i-1],list_num_hidden[i]) # 用relu 激活函数
        
        # 定义每一层的偏置
        layer["b"] = 0.1*np.ones([1,list_num_hidden[i]])
        layer["act_fun"]= list_act_funs[i]
        layer["de_act_fun"]= list_de_act_funs[i]
        layers.append(layer)
        
    return layers

def fead_forward(datas,layers):
    input_layers = []
    input_acfun = []
    for i in range(len(layers)):
        layer = layers[i]
        if i ==0:
            inputs = datas
            z = np.dot(inputs,layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
        else:
            inputs = h
            z = np.dot(inputs,layer["w"])+ layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
    return input_layers,input_acfun,h

# 进行参数更新    
def updata_wb(datas,labs,layers, loss_fun,de_loss_fun,alpha=0.01):
    N,D = np.shape(datas)
    # 进行前馈操作
    inputs,input_acfun,output = fead_forward(datas,layers)
    # 计算 loss
    loss = loss_fun(output,labs)
    #从后向前计算
    deltas0 = de_loss_fun(output,labs)
    # 从后向前计算误差
    deltas =[]
    for i in range(len(layers)):
        index = -i-1
        if i ==0:
            h = output
            z = input_acfun[index]
            delta = deltas0*layers[index]["de_act_fun"](z,h)
        else:
            h = inputs[index+1]
            z = input_acfun[index]
            # print(layers[index]["de_act_fun"](z,h)[1])
            delta = np.dot(delta,layers[index+1]["w"].T)*layers[index]["de_act_fun"](z,h)
        
        deltas.insert(0,delta)
    
    # 利用误差 对每一层的权重进行修成
    for i in range(len(layers)):
        # 计算 dw 与 db
        dw = np.dot(inputs[i].T,deltas[i])
        db = np.sum(deltas[i],axis=0,keepdims=True)
        # 梯度下降
        layers[i]["w"] = layers[i]["w"] - alpha*dw
        layers[i]["b"] = layers[i]["b"] - alpha*db
        
    return layers,loss
    
def test_accuracy(datas,labs_true,layers):
    _,_,output = fead_forward(datas,layers)
    lab_det = np.argmax(output,axis=1)
    labs_true = np.argmax(labs_true,axis=1) 
    N_error = np.where(np.abs(labs_true-lab_det)>0)[0].shape[0]
   
    error_rate = N_error/np.shape(datas)[0]
    return error_rate

def load_dataset_iris(file_data,N_train):
    # 数据读取
    datas = np.loadtxt(file_data,dtype = np.float64, delimiter = ',',usecols=(0,1,2,3))
    labs = np.loadtxt(file_data,dtype = str, delimiter = ',',usecols=(4))
    N,D = np.shape(datas)
    N_test = N-N_train
    unqiue_labs = np.unique(labs).tolist()
    
    dic_str2index={}
    dic_index2str={}
    for i in range(len(unqiue_labs)):
        lab_str = unqiue_labs[i]
        dic_str2index[lab_str] =i
        dic_index2str[i]=lab_str
    
    labs_onehot = np.zeros([N,len(unqiue_labs)])
    for i in range(N):
        labs_onehot[i,dic_str2index[labs[i]]]=1
    
    perm = np.random.permutation(N)
    index_train = perm[:N_train]
    index_test = perm[N_train:]

    data_train = datas[index_train,:]
    lab_train_onehot = labs_onehot[index_train,:]

    data_test = datas[index_test,:]
    lab_test_onehot = labs_onehot[index_test]

    return data_train,lab_train_onehot,data_test,lab_test_onehot,dic_index2str

def load_dataset_iris_my(file_data,N_train):
    # 数据读取
    # datas = np.loadtxt(file_data,dtype = np.float64, delimiter = ',',usecols=(0,1,2,3))
    # labs = np.loadtxt(file_data,dtype = str, delimiter = ',',usecols=(4))
    # print(labs.shape)
    
    t1 = 2*np.random.random(80) - 1
    t2 = 2*np.random.random(80) - 1
    x = np.column_stack((t1.reshape(-1,1),t2.reshape(-1,1)))
    y = np.zeros((80,1))
    
    t1 = np.linspace(0,2*np.pi,num=80) + np.random.random(80)/2
    t2 = 2*np.sin(t1)
    t3 = 2*np.cos(t1)
    t3 = t3 + np.random.random(80)/2
    x1 = np.hstack((t2.reshape(-1,1),t3.reshape(-1,1)))
    y1 = np.ones((80,1))

    datas = np.vstack((x,x1))
    labs = np.vstack((y,y1)).ravel()

    N,D = np.shape(datas)
    N_test = N-N_train
    unqiue_labs = np.unique(labs).tolist()

    dic_str2index={}
    dic_index2str={}
    for i in range(len(unqiue_labs)):
        lab_str = unqiue_labs[i]
        dic_str2index[lab_str] =i
        dic_index2str[i]=lab_str
    
    labs_onehot = np.zeros([N,len(unqiue_labs)])
    for i in range(N):
        labs_onehot[i,dic_str2index[labs[i]]]=1
    
    perm = np.random.permutation(N)
    index_train = perm[:N_train]
    index_test = perm[N_train:]

    data_train = datas[index_train,:]
    lab_train_onehot = labs_onehot[index_train,:]

    data_test = datas[index_test,:]
    lab_test_onehot = labs_onehot[index_test]

    return data_train,lab_train_onehot,data_test,lab_test_onehot,dic_index2str
    

if __name__=="__main__":
    file_data = "D:/pyprj/cv2/mydata/iris.data"
    # global layers
    # iris = datasets.load_iris()
    # iris_data = np.hstack((iris.data,iris.target.reshape(-1,1)))
    
    # data_train,lab_train_onehot,data_test,lab_test_onehot,dic_index2str =load_dataset_iris_my(file_data,100)
    data_train,lab_train_onehot,data_test,lab_test_onehot,dic_index2str =load_dataset_iris(file_data,100)

    N,dim_in = np.shape(data_train)
    # 定义网络结构
    list_num_hidden=[4,4,3]
    list_act_funs = [relu,relu,no_active] #[relu,relu,relu,relu,no_active]   #[relu,relu,no_active]  sigmod
    list_de_act_funs= [de_relu,de_relu,de_no_active] #[de_relu,de_relu,de_relu,de_relu,de_no_active]    #[de_relu,de_relu,de_no_active] de_sigmoid
    
    # 定义损失函数
    # loss_fun = loss_CE
    # de_loss_fun=de_loss_CE
    
    loss_fun = loss_L2
    de_loss_fun=de_loss_L2
    
    layers = bulid_net(dim_in,list_num_hidden,
          list_act_funs,list_de_act_funs)
    
   
    # 进行训练
    n_epoch = 200
    batchsize =4
    N_batch = N//batchsize
    for i in range(n_epoch):
        # 数据打乱
        rand_index  = np.random.permutation(N).tolist()
        # 每个batch 更新一下weight
        loss_sum =0
        for j in range(N_batch):
            index = rand_index[j*batchsize:(j+1)*batchsize]
            batch_datas = data_train[index]
            batch_labs = lab_train_onehot[index]
            layers,loss = updata_wb(batch_datas,batch_labs,layers,loss_fun,de_loss_fun,alpha=0.01)
            loss_sum = loss_sum+loss
            
        error = test_accuracy(data_train,lab_train_onehot,layers)
        print("epoch %d  error  %.2f%%  loss_all %.2f"%(i,error*100,loss_sum))
    
    #进行测试
    error = test_accuracy(data_test,lab_test_onehot,layers)
    print(error*100)
