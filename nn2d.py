import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy import integrate


small_num = 1.0e-15

torch.manual_seed(41)

# Definition of Neural Network

class Network(nn.Module):
    def __init__(self, nInput, nHidden1, nHidden2, nOutput):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(nInput, nHidden1)
        self.fc2 = nn.Linear(nHidden1, nHidden2)
        self.fc3 = nn.Linear(nHidden2, nOutput)

    def forward(self, x):#swish activation for 3 hidden layers and softmax for output
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = -1)
        return x

# Definition of network model

class Model():
    def __init__(self, nHidden1, nHidden2, u1):
        self.nStencil = 5        # No. of cells in the stencil
        self.nDelta   = 6        # No. of inputs in the delta layer
        self.nHidden1 = nHidden1 # No. of neurons in first hidden layer
        self.nHidden2 = nHidden2 # No. of neurons in second hidden layer
        self.nOutput  = 4        # No. of outputs (weights of the stencil)
        self.NN = Network(self.nDelta,self.nHidden1,self.nHidden2,self.nOutput)
        self.u1        = u1

        # Stencil ordering
        # Stencil 1: (i-1,j), (i,j), (i,j+1)
        # Stencil 2: (i+1,j), (i,j), (i,j+1)
        # Stencil 3: (i+1,j), (i,j), (i,j-1)
        # Stencil 4: (i-1,j), (i,j), (i,j-1)

        # Cell ordering
        # 0: (i,j)
        # 1: (i-1,j)
        # 2: (i+1,j)
        # 3: (i,j-1)
        # 4: (i,j+1)

        # Calculate delta

        self.delta = torch.tensor([u1[0]-u1[1],         # u_{i,j} - u_{i-1,j}
                                   u1[2]-u1[0],         # u_{i+1,j} - u_{i,j}
                                   u1[0]-u1[3],         # u_{i,j} - u_{i,j-1}
                                   u1[4]-u1[0],         # u_{i,j+1} - u_{i,j}
                                   u1[2]-2*u1[0]+u1[1],  # u_{i-1,j} - 2*u_{i,j} + u_{i+1,j}
                                   u1[4]-2*u1[0]+u1[3]]) # u_{i,j-1} - 2*u_{i,j} + u_{i,j+1}


        self.delta = torch.abs(self.delta) # Take absolute value of delta layer
        scale = torch.max(torch.tensor([self.delta[0], self.delta[1], self.delta[2], self.delta[3]])) # Scale the absolute value
        self.delta = self.delta/scale

        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=0.001)

        # Exact values

        # self.L = torch.tensor([-0.5])
        # self.R = torch.tensor([ 0.5])
        # self.B = torch.tensor([ 0.0])
        # self.T = torch.tensor([ 0.0])
        
        self.L = u1[5]
        self.R = u1[6]
        self.B = u1[7]
        self.T = u1[8]
        
    def residual(self):
        w = self.NN(self.delta)


        i = 0; im1 = 1; ip1 = 2
        jm1 = 3; jp1 = 4

        w0 = w[0]; w1 = w[1]; w2 = w[2]; w3 = w[3]

        # Stencil 1

        u_L0 = (1./2.)*self.u1[i] + (1./2.)*self.u1[im1]
        u_R0 = (3./2.)*self.u1[i] - (1./2.)*self.u1[im1]
        u_B0 = (3./2.)*self.u1[i] - (1./2.)*self.u1[jp1]
        u_T0 = (1./2.)*self.u1[i] + (1./2.)*self.u1[jp1]

        # Stencil 2

        u_L1 = (3./2.)*self.u1[i] - (1./2.)*self.u1[ip1]
        u_R1 = (1./2.)*self.u1[i] + (1./2.)*self.u1[ip1]
        u_B1 = (3./2.)*self.u1[i] - (1./2.)*self.u1[jp1]
        u_T1 = (1./2.)*self.u1[i] + (1./2.)*self.u1[jp1]

        # Stencil 3

        u_L2 = (3./2.)*self.u1[i] - (1./2.)*self.u1[ip1]
        u_R2 = (1./2.)*self.u1[i] + (1./2.)*self.u1[ip1]
        u_B2 = (1./2.)*self.u1[i] + (1./2.)*self.u1[jm1]
        u_T2 = (3./2.)*self.u1[i] - (1./2.)*self.u1[jm1]

        # Stencil 4

        u_L3 = (1./2.)*self.u1[i] + (1./2.)*self.u1[im1]
        u_R3 = (3./2.)*self.u1[i] - (1./2.)*self.u1[im1]
        u_B3 = (1./2.)*self.u1[i] + (1./2.)*self.u1[jm1]
        u_T3 = (3./2.)*self.u1[i] - (1./2.)*self.u1[jm1]

        # Combine all of them

        u_L = w0*u_L0 + w1*u_L1 + w2*u_L2 + w3*u_L3
        u_R = w0*u_R0 + w1*u_R1 + w2*u_R2 + w3*u_R3
        u_B = w0*u_B0 + w1*u_B1 + w2*u_B2 + w3*u_B3
        u_T = w0*u_T0 + w1*u_T1 + w2*u_T2 + w3*u_T3

        e_recon = (self.L - u_L)**2 + (self.R - u_R)**2 + (self.B - u_B)**2 + (self.T - u_T)**2
        e_weights = (0.25-w0)**2 + (0.25-w1)**2 + (0.25-w2)**2 +  (0.25-w3)**2

        # gamma_x = torch.abs(self.u1[im1]-2.*self.u1[i]+self.u1[ip1])/(torch.abs(self.u1[i]-self.u1[im1]) + torch.abs(self.u1[ip1]-self.u1[i]))
        # gamma_y = torch.abs(self.u1[jm1]-2.*self.u1[i]+self.u1[jp1])/(torch.abs(self.u1[i]-self.u1[jm1]) + torch.abs(self.u1[jp1]-self.u1[i]))
        
        
        gamma = torch.abs(self.u1[im1]+self.u1[ip1]+self.u1[jm1]++self.u1[jp1]-4.*self.u1[i]+self.u1[ip1])/(torch.abs(self.u1[i]-self.u1[im1]) + torch.abs(self.u1[ip1]-self.u1[i]) + torch.abs(self.u1[i]-self.u1[jm1]) + torch.abs(self.u1[jp1]-self.u1[i]))

        gamma = torch.pow(gamma,0.1) 

        #return  gamma*(e_recon) + 0*0.1*(1-gamma)*e_weights
        return  (e_recon)


    def train(self, epochs):

        lossTracker = []
        self.NN.train()
        for idx in range(epochs):

            loss = self.residual()
            if (idx%50 == 0):
                print(f"The loss at epoch {idx} is {loss.item()}")

            lossTracker.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"The loss at epoch {idx} is {loss.item()}")

        return lossTracker


    def predict(self):
        return self.NN(self.delta)



#%%

def tanh2D(x,y):
    beta = 2
    d = 0.25 #centre of transition
    return 0.5*(1+np.tanh(beta*(x+0*y-2*d)))

def sine2D(x,y):
    k = 1/2;
    return np.sin(k*np.pi*(x+y))

def plane(x,y):
    return x+y

def poly(x,y):
    return x**2

def recon(u1,h,w):
    
    ux_1 = u1[0] - u1[1] #stencil 1
    ux_2 = u1[2] - u1[0] #stencil 2
    ux_3 = u1[2] - u1[0] #stencil 3
    ux_4 = u1[0] - u1[1] #stencil 4
    
    uy_1 = u1[4] - u1[0] #stencil 1
    uy_2 = u1[4] - u1[0] #stencil 2
    uy_3 = u1[0] - u1[3] #stencil 3
    uy_4 = u1[0] - u1[3] #stencil 4
    
    m_x = ux_1*w[0] + ux_2*w[1] + ux_3*w[2] + ux_4*w[3]
    m_y = uy_1*w[0] + uy_2*w[1] + uy_3*w[2] + uy_4*w[3]
    
    u_iph = m_x*h/2 + u1[0]
    u_imh = m_x*(-h/2) + u1[0]
    u_jph = m_y*h/2 + u1[0]
    u_jmh = m_y*(-h/2) + u1[0]
    return u_iph.detach().numpy(), u_imh.detach().numpy(), u_jph.detach().numpy(), u_jmh.detach().numpy()
    
    
    
    
N = 3
h = 1
xmin = -N/2*h
xmax =  N/2*h
x = np.zeros(N)
y = np.zeros(N)
ymin = -N/2*h
ymax =  N/2*h
y = np.zeros(N)


for i in range(0,N):
    x[i] = xmin + (float(i) + 0.5)*h
    y[i] = ymin + (float(i) + 0.5)*h
    
u = np.zeros((N,N), dtype=np.float32)
#%%

func = tanh2D

for i in range(0,N):
    for j in range(0,N):
        u[i,j], error = (integrate.dblquad(func, x[i]-h/2, x[i]+h/2,
                                           y[j]-h/2 , y[j]+h/2))

u = u/(h*h)

uL_e = func(-0.5*h,0)
uR_e = func( 0.5*h,0)
uB_e = func(0,-0.5*h)
uT_e = func(0, 0.5*h)

u1 = torch.tensor(np.float32([u[1][1], u[0][1], u[2][1], u[1][0], u[1][2], uL_e, uR_e, uB_e, uT_e])) # Stencil


nHidden1 = 3
nHidden2 = 2

WENO3_NN = Model(nHidden1,nHidden2,u1)

WENO3_NN.residual()

WENO3_NN.train(40000)

w = WENO3_NN.predict()

print('Predicted weights = ', w)

u_iph, u_imh, u_jph, u_jmh=recon(u1,h,w)

Err = np.array([abs(u_iph-func(0.5*h,0)), abs(u_imh-func(-0.5*h,0)), abs(u_jph-func(0,0.5*h)), abs(u_jmh-func(0,-0.5*h))]).reshape(-1)

error = np.sum(Err)
