# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@authors:   Jamie Taylor (CUNEF) 
            Manuela Bastidas (UPV/EHU)
            https://www.mathmode.science/ 
'''

# This code presents a simple implementation of the Deep Fourier Method, 
# which can be seen as an specific instance of Variational Physics-Informed 
# Neural Networks (VPINNs).   -- <50 lines of VPINNS--

# In this 1D example, we utilize Keras_core for constructing neural networks 
# and TF in the backend. 
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

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras

# Set the random seed
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
keras.backend.set_floatx(dtype)


# =============================================================================
# 
#          Source code - DFR H01 1D 
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
    l2 = keras.layers.Dense(1, activation=activation, dtype=dtype)(l1)

    # A cut-off layer to impose the boundary conditions (dirichlet 0)
    output = cutoff_layer()([xvals,l2])

    u_model = keras.Model(inputs = xvals, outputs = output, name='u_model')

    # Print the information of the model u
    u_model.summary()

    return u_model


## A custom Keras layer for enforcing boundary conditions (BC)
class cutoff_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Initializes the cutoff layer.
        
        """
        super(cutoff_layer, self).__init__()

    def call(self, inputs):
        """
        Applies the cutoff operation to the last layer to impose boundary conditions.

        Args:
            inputs: A tuple containing two tensors, x and u, where x represents 
                    spatial coordinates and u is the predicted solution.

        Returns:
            keras.Tensor: A tensor representing the modified solution after 
                          applying the cutoff.
        """
        x, u = inputs
        # The cutoff condition: u is zero when x = 0 or pi
        cut = x * (x - np.pi)
        # Compute the element-wise product of cut and u
        # This effectively enforces Dirichlet boundary condition u = 0 at x = pi
        return keras.ops.einsum('ij,ij->i', cut, u)


## Deep Fourier Residual (DFR) loss function as a custom Keras layer
class loss(keras.layers.Layer):
    def __init__(self, u_model, n_pts, n_modes, f, **kwargs):
        """
        Initializes the DFR loss layer with provided parameters.

        Args:
            u_model (keras.Model): The neural network model for the approximate 
                                    solution.
            n_pts (int): Number of integration points.
            n_modes (int): Number of Fourier modes.
            f (function): Source - RHS of the PDE 
            
            kwargs: Additional keyword arguments.
        """
        super(loss, self).__init__()
        self.u_model = u_model

        # The domain is (0,pi) by default, include as input is the domain change
        b = np.pi
        a = 0
        
        # Generate integration points
        hi_pts = np.linspace(a,b, n_pts+1)

        diff = keras.ops.abs(hi_pts[1:] - hi_pts[:-1])
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

        #self.f = f(self.pts)
	# The source part (RHS) of the formulation 
        self.FT_low = keras.ops.einsum("ji,i->j", DST, f(self.pts))
        
    def call(self, inputs):
        """
        Computes the Deep Fourier Regularization (DFR) loss.

        Args:
            inputs: The input data (dummy).

        Returns:
            keras.Tensor: The DFR loss value.
        """
        ## Evaluate u and its derivative at integration points
        
        # Evaluate u and its derivative at integration points
        ## Persistent True not necessary because it only evaluates u'(once)
        with tf.GradientTape() as t1:
            t1.watch(self.pts)
            u = self.u_model(self.pts)
        du = t1.gradient(u,self.pts)

        ## Take appropriate transforms of each component
        FT_high = keras.ops.einsum("ji,i->j", self.DCT, du)

        ## Add and multiply by weighting factors
        FT_tot = (FT_high + self.FT_low) * self.coeffs

        ## Return sum of squares loss
        return keras.ops.sum(FT_tot**2)


## Create a loss model
def make_loss_model(u_model, n_pts, n_modes, f):
    """
    Constructs a loss model for Deep Fourier Residual method (DFR).

    Args:
        u_model (keras.Model): The neural network model for the approximate solution.
        n_pts (int): Number of integration points.
        n_modes (int): Number of Fourier modes.

    Returns:
        keras.Model: A model with the DFR loss function.
    """
    xvals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    
    # Compute the DFR loss using the provided neural network and 
    # integration parameters
    output = loss(u_model, n_pts, n_modes, f)(xvals)

    # Create a Keras model for the DFR loss
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
n_pts = 100
# Number of Fourier modes
n_modes = 10
# Number of training iterations
iterations = 1000

# Initialize the neural network model for the approximate solution
u_model = make_model(neurons=nn, n_layers=nl)

# Big model including the Deep Fourier Regularization (DFR) loss
loss_model = make_loss_model(u_model, n_pts, n_modes, f_rhs)

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



