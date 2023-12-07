# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@authors:   Jamie Taylor (CUNEF) 
            Manuela Bastidas (UPV/EHU)
            https://www.mathmode.science/ 
'''

# This code presents a simple implementation of Physics-Informed 
# Neural Networks (PINNs) as a collocation method. -- <50 lines of PINNS--

# In this 1D example, we utilize Keras_core for constructing neural networks 
# and TF in the backend. 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras

# Set the random seed
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
keras.backend.set_floatx(dtype)


# =============================================================================
# 
#          Source code - PINNs H01 1D 
#
# =============================================================================


## Define an approximate solution (u_nn): A neural network model
def make_model(neurons, n_layers, activation='tanh'):
    
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
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

    ## ---------------
    #  The dense layers
    ## ---------------
    
    # First layer
    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xvals)
    for l in range(n_layers-2):
        # Hidden layers
    	l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    # Last layer
    output = keras.layers.Dense(1, activation=activation, dtype=dtype)(l1)

    u_model = keras.Model(inputs = xvals, outputs = output, name='u_model')

    # Print the information of the model u
    u_model.summary()

    return u_model

##PINNs loss function ( loss into layer )
class loss(keras.layers.Layer):
    def __init__(self,u_model,n_pts,f,**kwargs):
    
        """
        Initializes the PINNS loss layer with provided parameters.
    
        Args:
            u_model (keras.Model): The neural network model for the approximate 
                                    solution.
            n_pts (int): Number of integration points.
            f (function): Source - RHS of the PDE 
            
            kwargs: Additional keyword arguments.
        """
        super(loss, self).__init__()
    
        self.u_model = u_model
        self.n_pts = n_pts
        self.f = f
            
    def call(self,inputs):
        
        """
        Computes the collocation - PINNs loss.

        Args:
            inputs: The input data (dummy).

        Returns:
            keras.Tensor: The loss value.
        """
        
        # Generate random integration points
        x = tf.random.uniform([self.n_pts],dtype=dtype,maxval=np.pi)
        
        # Take gradients (tf based part)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                u = self.u_model(x,training=True)
            dux = t2.gradient(u,x)
        duxx = t1.gradient(dux,x)
        
        ## u''-sin(2x)=0 is eq, PDE loss
        error_PDE = keras.ops.mean((duxx-self.f(x))**2)
        
        ##Dirichlet BC loss
        bc = self.u_model(np.array([np.pi]))**2 + self.u_model(np.array([0.]))**2
        return error_PDE + bc
    

## Create a loss model
def make_loss_model(u_model, n_pts, f):
    """
    Constructs a loss model for PINNs.

    Args:
        u_model (keras.Model): The neural network model for the approximate solution.
        n_pts (int): Number of integration points.

    Returns:
        keras.Model: A model with the collocation-based loss function.
    """
    xvals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    
    # Compute the loss using the provided neural network and 
    # integration parameters
    output = loss(u_model, n_pts, f)(xvals)

    # Create a Keras model for the loss
    loss_model = keras.Model(inputs=xvals, outputs=output)
    
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
   return -4*keras.ops.sin(2 * x)

# Number of neurons per hidden layer in the neural network
nn = 10
# Number of hidden layers 
nl = 4
# Number of integration points
n_pts = 1000
# Number of training iterations
iterations = 1000

# Initialize the neural network model for the approximate solution
u_model = make_model(neurons=nn, n_layers=nl)

# Big model including the  loss
loss_model = make_loss_model(u_model, n_pts, f_rhs)

# Optimizer (Adam optimizer with a specific learning rate)
optimizer = keras.optimizers.Adam(learning_rate=10**-3)

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
    return keras.ops.sin(2 * x)

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



