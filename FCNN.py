import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt

##### IMPORT CIFAR10 #####
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainvalset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(trainvalset, [45000, 5000])

batch_size = 50
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)


train_X = []
train_Y = []
for image, label in trainloader:
  train_X.append(image.reshape(batch_size, 3072).numpy())
  t = np.zeros((batch_size,10))
  for i in range(batch_size):
    t[i,label[i]] = 1.
  train_Y.append(t)
  
val_X = []
val_Y = []
for image, label in valloader:
  val_X.append(image.reshape(batch_size, 3072).numpy())
  t = np.zeros((batch_size,10))
  for i in range(batch_size):
    t[i,label[i]] = 1.
  val_Y.append(t)
  
test_X = []
test_Y = []
for image, label in testloader:
  test_X.append(image.reshape(batch_size, 3072).numpy())
  t = np.zeros((batch_size,10))
  for i in range(batch_size):
    t[i,label[i]] = 1.
  test_Y.append(t)

##### FULLY CONNECTED #####
class NeuralNetwork:
  
  def __init__(self, layer_dimensions,alpha,batch):
    self.layers = len(layer_dimensions)
    self.W = [ np.random.uniform(-1./np.sqrt(ld[0]),1./np.sqrt(ld[0]),ld) for ld in layer_dimensions ]
    self.b =  [ np.random.uniform(-1./np.sqrt(ld[0]),1./np.sqrt(ld[0]),(1,ld[1])) for ld in layer_dimensions ]
    self.alpha = alpha
    self.batch = batch
    
  def affineForward(self,A,W,b):  #Aw+b#
    out = np.dot(A,W) + b
    deriv = A
    return out, deriv
   
  def activationForward(self,A): ## ReLU
    A[A <= 0.] = 0.
    deriv  = (A > 0.).astype(int)
    return A, deriv

  def softMax(self,A): #For cross-entropy
    A = A - np.max(A,axis=1).reshape(self.batch,1)
    exp = np.exp(A)
    out = exp/(np.sum(exp,axis=1).reshape(self.batch,1))
    
    d1 = out*(1.0-out)
    d1 = d1.reshape(self.batch,10,1)
    d1 = np.repeat(d1,10,axis=2)
    d2 = out.reshape(self.batch,10,1) * out.reshape(self.batch,1,10)
    d2 = -d2
    I = np.identity(10)
    I = np.repeat(I.reshape(1,10,10),self.batch,axis=0)
    d1 = d1 * I
    I2 = np.ones((self.batch,10,10))
    d2 = d2 * (I2-I)
    deriv = d1 + d2
    return out, deriv
  
  def costFunction(self,AL,y): # cross-entropy
    eps = 1e-20
    lg = np.log(AL+eps)
    out = -np.average(np.multiply(y, lg),axis=1)
    deriv = -np.divide(y,AL,out=np.zeros_like(y),where=AL!=0)
    return out, deriv
     
  def Forward(self,A):
    LO, LD, AO, AD = [],[],[],[]
    for i in range(self.layers):
      layer_out, layer_deriv = 0,0
      if i==0:
        layer_out, layer_deriv = self.affineForward(A,self.W[i],self.b[i])
        activ_out, activ_deriv = self.activationForward(layer_out)
        LO.append(layer_out)
        LD.append(layer_deriv)
        AO.append(activ_out)
        AD.append(activ_deriv)
      elif i == (self.layers - 1):
        layer_out, layer_deriv = self.affineForward(AO[i-1],self.W[i],self.b[i])
        LO.append(layer_out)
        LD.append(layer_deriv)
      else:
        layer_out, layer_deriv = self.affineForward(AO[i-1],self.W[i],self.b[i])
        activ_out, activ_deriv = self.activationForward(layer_out)
        LO.append(layer_out)
        LD.append(layer_deriv)
        AO.append(activ_out)
        AD.append(activ_deriv)
    SM_out, SM_deriv = self.softMax(LO[-1])
    return SM_out , SM_deriv, LO,LD,AO,AD
  
  def Backward(self,A,y):
    sm_o, sm_d, layers_o, layers_d, activ_o, activ_d = self.Forward(A) #activate forward pass
    err_o, err_d = self.costFunction(sm_o,y) #compute cost function
    sigma, w_grad, b_grad = [0.]*self.layers, [0.]*self.layers, [0.]*self.layers #create zeroed out arrays for computing backprop
    for j in range(self.layers): #propagate the derivative
      i = self.layers-1-j
      if i==self.layers-1:
        err_d = err_d.reshape(self.batch,1,10)
        sigma[i] = np.matmul(err_d, sm_d)
        sigma[i] = sigma[i].reshape(self.batch,10)
        s = sigma[i][...,np.newaxis].transpose((0,2,1)) #(50,10)
        a = activ_o[-1][...,np.newaxis] #(50,256)
        w_grad[i] = s*a
        b_grad[i] = sigma[i]
      elif i==0:
        sigma[i] = (sigma[i+1] @ self.W[i+1].T)*activ_d[i]
        s = sigma[i][...,np.newaxis].transpose((0,2,1))
        a = A[...,np.newaxis]
        w_grad[i] = s*a
        b_grad[i] = sigma[i]
      else:
        sigma[i] = (sigma[i+1] @ self.W[i+1].T)*activ_d[i]
        s = sigma[i][...,np.newaxis].transpose((0,2,1))
        a = activ_o[i-1][...,np.newaxis]
        w_grad[i] = s*a
        b_grad[i] = np.average(sigma[i], axis=0)
    return np.argmax(sm_o,axis=1),err_d,w_grad,b_grad, np.average(err_o)
  
  def update(self, w_grad, b_grad):      
      for i in range(self.layers):
        self.W[i] -= self.alpha*np.average(w_grad[i], axis=0)
        self.b[i] -= self.alpha*np.average(b_grad[i], axis=0)
        
  def predict(self,A):
    prob,_,_,_,_,_ = self.Forward(A)
    return np.argmax(prob, axis=1)
  
  def Train(self, tX, tY, vX, vY, epoch):
    for i in range(1,epoch+1):
      print("this is epoch {} ".format(i))
      train_correct = 0
      train_loss = 0.
      for x,y in zip(tX,tY):
        pred,_,dw,db,loss = self.Backward(x,y)
        self.update(dw,db)
        train_loss += loss
        y_l = np.argmax(y,axis=1)
        for i in range(self.batch):
          if pred[i] == y_l[i]:
            train_correct += 1
      
      accu = float(train_correct)/float(50*len(tX))
      print("Training Accuracy {0:.2f}".format(accu))
      print("Loss: {0:.2f}".format(train_loss/float(len(tX))))
      
      val_correct = 0
      for x,y in zip(vX,vY):
        pred = self.predict(x)
        y_l = np.argmax(y,axis=1)
        for i in range(self.batch):
          if pred[i] == y_l[i]:
            val_correct += 1
      val_accu = float(val_correct)/float(5000)
      print("Validation Accuracy: {0:.3f}".format(val_accu))
      
  def Test(self, tX, tY):
    test_correct = 0
    for x,y in zip(tX,tY):
        pred = self.predict(x)
        y_l = np.argmax(y,axis=1)
        for i in range(self.batch):
          if pred[i] == y_l[i]:
            test_correct += 1
    
    print(test_correct)
    accu = float(test_correct)/float(50*len(tX))
    print("Test Accuracy {0:.2f}".format(accu))

   
      

layer_dim = [[3072,256],[256,10]]
batch = 50
alpha = 0.01
NN = NeuralNetwork(layer_dim,alpha,batch)

train_X = []
train_Y = []
for image, label in trainloader:
  train_X.append(image.reshape(batch, 3072).numpy())
  t = np.zeros((batch,10))
  for i in range(batch):
    t[i,label[i]] = 1.
  train_Y.append(t)

print(len(train_X))
val_X = []
val_Y = []
for image, label in valloader:
  val_X.append(image.reshape(batch, 3072).numpy())
  t = np.zeros((batch,10))
  for i in range(batch):
    t[i,label[i]] = 1.
  val_Y.append(t)
  
test_X = []
test_Y = []
for image, label in testloader:
  test_X.append(image.reshape(batch, 3072).numpy())
  t = np.zeros((batch,10))
  for i in range(batch):
    t[i,label[i]] = 1.
  test_Y.append(t)

NN.Train(train_X,train_Y,val_X,val_Y,15)
NN.Test(test_X,test_Y)

### Test Image ###
ex1 = np.transpose(test_X[0][0].reshape(3,32,32), (1,2,0))
ex1 = ex1/2 + 0.5
plt.imshow(ex1, interpolation='nearest')
pred=NN.predict(test_X[0])
print("Predicted label:", pred[0])
print("True Label: {}".format(np.argmax(test_Y[0][0])))

ex2 = np.transpose(test_X[1][0].reshape(3,32,32), (1,2,0))
ex2 = ex2/2 + 0.5
plt.imshow(ex2, interpolation='nearest')
pred=NN.predict(test_X[1])
print("Predicted label:", pred[0])
print("True Label: {}".format(np.argmax(test_Y[1][0])))

