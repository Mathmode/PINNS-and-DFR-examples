# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@authors:   Jamie Taylor (CUNEF) 
            Manuela Bastidas (UPV/EHU)
            https://www.mathmode.science/ 
'''

# This code presents a simple implementation of the Deep Fourier Method, 
# which can be seen as an specific instance of Variational Physics-Informed 
# Neural Networks (VPINNs). 

# In this 1D example, we utilize tf.keras_core for constructing neural networks 
# and JAX for automatic differentiation. 
# This code serves as a basic introduction to the Deep Fourier Method and 
# can be extended for more complex computational physics applications. 

# This code uses the following weak-residual based loss:
    # int (grad u ).(grad v) - int f.v = R(u)

# Consult relevant literature and documentation for a deeper understanding
# of VPINNs and their wide range of applications.
# https://www.sciencedirect.com/science/article/abs/pii/S0045782522008064

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)


# =============================================================================
# 
#          Source code - DFR H01 1D 
#
# =============================================================================


## Define an approximate solution (u_nn): A neural network model
def make_model(neurons, neurons_last, n_layers, activation='tanh'):
    
    """
    Creates a neural network model to approximate the solution of 
        int (grad u ).(grad v) - int f.v = 0

    Args:
        neurons (int): The number of neurons in each hidden layer.
        activation (str, optional): Activation function for hidden layers.

    Returns:
        keras.Model: A neural network model for the approximate solution.
    """
    
	# The input
    xvals = tf.keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

    ## ---------------
    #  The dense layers
    ## ---------------
    
    # First layer
    l1 = tf.keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xvals)
    for l in range(n_layers-2):
        # Hidden layers
    	l1 = tf.keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    # Last layer
    l1 = tf.keras.layers.Dense(neurons_last, activation=activation, dtype=dtype)(l1)
    
    # The cutoff layer impose the homogeneous dirichlet boundary conditions 
    # on the nodes of the last hidden layer of the network
    l2 = cutoff_layer()([xvals,l1])
    
    # This is the model until the last layer
    u_model_LS = tf.keras.Model(inputs = xvals,outputs = l2,name='u_model_LS')
    
    u_model_LS.summary()
    
    # the output layer is cutomized (NO BIAS)
    output = linear_last_layer(neurons_last,dtype=dtype)(l2)

    # Create the model
    u_model = tf.keras.Model(inputs = xvals,outputs = output,name='u_model')
    
    # Print the information of the model u
    u_model.summary()
    
    return u_model, u_model_LS


## A custom tf.keras layer for enforcing boundary conditions (BC)
class cutoff_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Initializes the cutoff layer.
        
        """
        super(cutoff_layer, self).__init__()

    def call(self, inputs):
        """
        Applies the cutoff operation to the last hidden layer to impose boundary conditions.

        Args:
            inputs: A tuple containing two tensors, x and u, where x represents 
                    spatial coordinates and u is the values of the neurons in the last layer.

        Returns:
            tf.keras.Tensor: A tensor representing the modified solution after 
                          applying the cutoff.
        """
        x, N = inputs
        # The cutoff condition: u is zero when x = 0 or pi
        cut = x * (x - np.pi)
        # Compute the element-wise product of cut and the neurons on the last hidden layer
        # This effectively enforces Dirichlet boundary condition u = 0 at x = pi
        return tf.einsum("ij,ik->ik",cut,N) 
    

# The last layer - output (customized)
class linear_last_layer(tf.keras.layers.Layer):
    def __init__(self,nn_last,dtype='float64',**kwargs):
        super(linear_last_layer,self).__init__()
        
        # Standard initialization of the weights (TF)
        pweight = tf.random.uniform([nn_last],minval=-(6/(1+nn_last))**0.5,
                                    maxval=(6/(1+nn_last))**0.5,dtype =dtype)
        
        # The weights of the last layer (modified in the LS)
        # TRAIN = True -> NO LS
        # TRAIN = False -> LS
        self.vars = tf.Variable(pweight,trainable=False,dtype=dtype)
    
    def call(self,inputs):
        pweights = self.vars #[:-1]
        #bias = self.vars[-1]
        return tf.einsum("i,ji->j",pweights,inputs)#+bias

    

## Deep Fourier Residual (DFR) loss function as a custom tf.keras layer
class loss(tf.keras.layers.Layer):
    def __init__(self, u_model, u_model_LS, n_pts, n_modes, f, regul=1e-5, **kwargs):
        """
        Initializes the DFR loss layer with provided parameters.

        Args:
            u_model (tf.keras.Model): The neural network model for the approximate 
                                    solution.
            u_model_LS (tf.keras.Model): The last hidden layer
            n_pts (int): Number of integration points.
            n_modes (int): Number of Fourier modes.
            f (function): Source - RHS of the PDE 
            regul (float): The regularization parameter for the LS system
            
            kwargs: Additional keyword arguments.
        """
        super(loss, self).__init__()
        self.u_model = u_model
        self.u_model_LS = u_model_LS
        
        # Number of neurons last hidden layer
        self.nn = self.u_model.layers[-2].output_shape[1]
        self.regul = regul

        # The domain is (0,pi) by default, include as input is the domain change
        b = np.pi
        a = 0
         
        # Generate integration points
        hi_pts = np.linspace(a, b, n_pts + 1)

        diff = tf.abs(hi_pts[1:] - hi_pts[:-1])
        self.pts = hi_pts[:-1] + diff / 2

        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([((np.pi**2 * k**2) / (b - a)**2)**-0.5 
                                for k in range(1, n_modes + 1)])

        ##----------
        # NOTE: The DST and DCT are computed explicitly here
        # To improve: Use the fast DST of python 
        ##---------
        
        # Matrix for Sine transform
        DST = np.array([[np.sqrt(2 / (b - a)) * np.sin(np.pi * k * (self.pts[i] - a) / (b - a)) * diff[i] 
                              for i in range(len(self.pts))] for k in range(1, n_modes + 1)])

        # Matrix for Cosine transform
        self.DCT = np.array([[np.sqrt(2 / (b - a)) * (k * np.pi / (b - a)) * np.cos(np.pi * k * (self.pts[i] - a) / (b - a)) * diff[i] 
                              for i in range(len(self.pts))] for k in range(1, n_modes + 1)])
        
        self.vec_B = tf.einsum("ji,i,j->j",DST,f(self.pts),self.coeffs)
        
        # Number of modes
        self.n_modes = n_modes
        
    def call(self, inputs):
        """
        Computes the Deep Fourier Regularization (DFR) loss.

        Args:
            inputs: The input data (dummy).

        Returns:
            tf.keras.Tensor: The DFR loss value.
        """
        ## Evaluate u (Last hidden layer) and the derivatives
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.pts)
            # Last hidden layer evaluation
            u = tf.unstack(self.u_model_LS(self.pts),axis=-1)
        
        # Derivative of the last hidden layer functions
        # Improvements: (i) Jacobian (not the usual, it is vectorial function and scalar evaluation)
        # (ii) batch_jacobian: Imply reshaping pts and (iii) use matv instead of for 
        du = tf.stack([t1.gradient(u[i],self.pts) for i in range(self.nn)])
        del t1 

        ## Take appropriate transforms of each component
        # The LS matrix
        mat_A = tf.einsum("ji,ki,j->jk",self.DCT,du,self.coeffs)

        # The LS solution: The regularization is very important/sensitive here!!
        solution_w0 = tf.squeeze(tf.linalg.lstsq(mat_A,
                                                 -tf.reshape(self.vec_B,(self.n_modes,1)), 
                                                 l2_regularizer=self.regul))
         
        # Assign the weights
        self.u_model.layers[-1].vars.assign(solution_w0)
        
        #w_last = self.u_model.layers[-1].vars
        # Ax : The RHS of the weak/ultraweak formulation
        FT_high = tf.einsum("ji,i->j",mat_A,solution_w0)
        
        # The value of the loss function
        return tf.reduce_sum((FT_high+self.vec_B)**2)


## Create a loss model
def make_loss_model(u_model, u_model_LS, n_pts, n_modes, f):
    """
    Constructs a loss model for Deep Fourier Residual method (DFR).

    Args:
        u_model (tf.keras.Model): The neural network model for the approximate solution.
        n_pts (int): Number of integration points.
        n_modes (int): Number of Fourier modes.

    Returns:
        tf.keras.Model: A model with the DFR loss function.
    """
    xvals = tf.keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    
    # Compute the DFR loss using the provided neural network and 
    # integration parameters
    output = loss(u_model, u_model_LS, n_pts, n_modes, f)(xvals)

    # Create a tf.keras model for the DFR loss
    loss_model = tf.keras.Model(inputs=xvals, outputs=output)
    
    return loss_model


def tricky_loss(y_pred, y_true):
    """
    A placeholder loss function that can be replaced as needed.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        float: The loss value.
    """
    # This is a placeholder loss function that can be substituted with a 
    # custom loss if required.
    return y_true

# =============================================================================
# 
#          Example 1 - Inputs
#
# =============================================================================

# PDE RHS 
def f_rhs(x):
   return -4*tf.sin(2 * x)

# Number of neurons per hidden layer in the neural network
nn = 10
# Number of neurons of the last hidden layer 
nn_last = 20

# Number of hidden layers 
nl = 4
# Number of integration points
n_pts = 100
# Number of Fourier modes
n_modes = 10
# Number of training iterations
iterations = 10000

# Initialize the neural network model for the approximate solution
u_model, u_model_LS = make_model(neurons=nn, neurons_last=nn_last, n_layers=nl)

# Big model including the Deep Fourier Regularization (DFR) loss
loss_model = make_loss_model(u_model, u_model_LS, n_pts, n_modes, f_rhs)

# Optimizer (Adam optimizer with a specific learning rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

# Compile the loss model with a custom loss function (tricky_loss)
loss_model.compile(optimizer=optimizer, loss=tricky_loss)

# Train the model using a single training data point ([1.], [1.]) for a 
# specified number of epochs (iterations)
history = loss_model.fit(np.array([1.]), np.array([1.]), epochs=iterations)


## ----------------------------------------------------------------------------
#   Plot the results
## ----------------------------------------------------------------------------

from matplotlib import rcParams


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['legend.fontsize'] = 17
rcParams['mathtext.fontset'] = 'cm' 
rcParams['axes.labelsize'] = 19

# Exact solution
def exact_u(x):
    return tf.sin(2 * x)

# Generate a list of x values for visualization
xlist = np.array([np.pi/1000 * i for i in range(1000)])

## ---------
# SOLUTION
## ---------

fig, ax = plt.subplots()
# Plot the approximate solution obtained from the trained model
plt.plot(xlist, u_model(xlist), color='b')
plt.plot(xlist, exact_u(xlist), color='m')

plt.legend(['u_approx', 'u_exact'])

ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
plt.show()


## ---------
# Loss evolution
## ---------

fig, ax = plt.subplots()
# Plot the approximate solution obtained from the trained model
plt.plot(history.history['loss'], color='r')

ax.set_xscale('log')
ax.set_yscale('log')

plt.legend(['loss'])

ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
plt.show()



