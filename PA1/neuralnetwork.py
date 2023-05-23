import numpy as np
import activations as ac


# Weight initialization and Building the network


def glorot_initialization( inputsize, outputsize):
    wij = np.sqrt(6/(inputsize+outputsize))
    return np.random.uniform(-wij, wij, (outputsize, inputsize))


def initialize_weights(sizes):
    parameters={}
    for i in range(len(sizes)-1):
        parameters["W" + str(i+1)]=glorot_initialization(sizes[i],sizes[i+1])
        parameters["b" + str(i+1)]=np.zeros((sizes[i+1],1))
    return parameters

#################   Forward pass   ####################################################


#  sizes=List of nodes in each layer including the input and output layer. For baseline model sizes=[784,500,250,100,10]
#  X=input image flatten to a vector



def forward_pass(sizes,X): 
    
    parameters=initialize_weights(sizes)
    num_hidden=len(sizes)-2
    activation_matrix={}
    if X.ndim == 1:
        X = X[:, np.newaxis] 
    Hmat={"h0":X}
    for i in range(1,num_hidden+2):
        Wl=parameters["W"+str(i)]
        bl=parameters["b"+str(i)]
        #print(Wl.shape)
        #print(bl.shape)
        
        hprev=Hmat["h"+str(i-1)]
        #print(hprev.shape)
        al=np.matmul(Wl,hprev)+ bl
        activation_matrix["a"+str(i)]=al
        if(i!=num_hidden +1):
            hl=ac.sigmoid(al)
        elif (i==num_hidden+1):
            hl=ac.softmax(al)
        Hmat["h"+str(i)]=hl

    yhat = Hmat["h"+str(num_hidden+1)]
    return yhat,activation_matrix,Hmat

#################  Back-prop  ##########################################


def creategrads(sizes): #this function initializes the gradient's matrix as zero
    num_hidden=len(sizes)-2
    inputsize=sizes[0]
    
    grads = {"dh0":np.zeros((inputsize,1)),
            "da0":np.zeros((inputsize,1))}
    for i in range(1, num_hidden+2):
        grads["dW" + str(i)] = np.zeros((sizes[i], sizes[i-1]))
        grads["db" + str(i)] = np.zeros((sizes[i],1))
        grads["da" + str(i)] = np.zeros((sizes[i],1))
        grads["dh" + str(i)] = np.zeros((sizes[i],1))
        
       
    return grads

#Gradient of activations functions
def grad_sigmoid(z):
    return (ac.sigmoid(z))*(1 - ac.sigmoid(z))

def grad_tanh(z):
    return (1 - (np.tanh(z))**2)

def grad_relu(z):
    return (z>0)*(np.ones(np.shape(z))) + (z<0)*(0.01*np.ones(np.shape(z)))


def back_prop(H, A, parameters, sizes, Y, Yhat, loss, activation):
    inputsize=sizes[0]
    num_hidden=len(sizes)-2
    grad_one_eg = creategrads( sizes)
    A["a0"] = np.zeros((inputsize,1))
    
    
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Yhat.ndim == 1:
        Yhat = Yhat[:, np.newaxis]
    
    if loss == "ce":
        grad_one_eg["da" + str(num_hidden + 1)] = Yhat - Y
    elif loss == "sq":
        grad_one_eg["da" + str(num_hidden + 1)] = (Yhat - Y)*Yhat - Yhat*(np.dot((Yhat-Y).T, Yhat))
   
    for i in np.arange(num_hidden + 1, 0, -1):
        
            grad_one_eg["dW" + str(i)] = np.matmul(grad_one_eg["da" + str(i)], (H["h" + str(i-1)]).T)
            grad_one_eg["db" + str(i)] = grad_one_eg["da" + str(i)]

            grad_one_eg["dh" + str(i-1)] = np.dot((parameters["W" + str(i)]).T, grad_one_eg["da" + str(i)])

            if activation == "sigmoid":
                derv = grad_sigmoid(A["a" + str(i-1)])
            elif activation == "tanh":
                derv = grad_tanh(A["a" + str(i-1)])
            elif activation == "relu":
                derv = grad_relu(A["a" + str(i-1)])
            if derv.ndim == 1:
                derv = derv[:, np.newaxis]

            grad_one_eg["da" + str(i-1)] = (grad_one_eg["dh" + str(i-1)])*derv

    return grad_one_eg


############ Training ##############


def training_nn(train_data,train_label,sizes,batch_size,learning_rate,epoch_size,acti):
    epoch=0
    updatepoint=0
    batch_data = {}
    epoch_data = []

    while(epoch<epoch_size):
        parameters=initialize_weights(sizes)
        gradient_matrix=creategrads(sizes)
        step=0
        for j in range(0,1000):
            x=train_data[j,:].flatten()
            y=train_label[j,:]

            yhat,activation_mat,hmat=forward_pass(sizes,x)
            grad_parameters=back_prop(hmat,activation_mat,parameters,sizes,y,yhat,'ce',acti)

            for key in gradient_matrix:
                gradient_matrix[key]=gradient_matrix[key] + grad_parameters[key]

            updatepoint=updatepoint+1

            if(updatepoint%batch_size==0):
                for key in parameters:
                    parameters[key]=parameters[key]-learning_rate*gradient_matrix['d'+key]
                gradient_matrix=creategrads(sizes)
                #step=step+1
                train_error_batch=calculate_performance(y,yhat,'ce')
                batch_data[(epoch,updatepoint//batch_size)]=train_error_batch

        train_error_epoch=calculate_performance(y,yhat,'ce')
        epoch_data.append([epoch,train_error_epoch])
        epoch=epoch+1
    return epoch_data,batch_data,parameters


            
    
def cross_entropy_loss(y,yhat):
    loss = -np.sum(y*np.log(yhat))
    return loss/float(yhat.shape[0])

def squared_error_loss(y,yhat):
    return 0.5*np.sum((y-yhat)**2)


def calculate_performance(y,yhat,loss):
    if(loss=='ce'):
        loss= cross_entropy_loss(y,yhat)
    else:
        loss= squared_error_loss(y,yhat)
    return loss






