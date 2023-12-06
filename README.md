# PINNs & Deep Fourier Residal Benchmark (Keras Core) - JAX & TensorFlow.

PINNs & Deep Fourier Residual Benchmark (Keras Core) - JAX & TensorFlow. Benchmarking for efficient PDE solving with NN.

Welcome to the "PINNs and Deep Fourier Residual Method Benchmark" repository, a collaborative effort by the MATHMODE group (https://www.mathmode.science/). Our goal is to provide a comprehensive benchmarking platform for different basic implementations of Physics-Informed Neural Networks (PINNs) and the Deep Fourier Residual Method (DFR), all based on the versatile Keras Core framework.

https://www.sciencedirect.com/science/article/abs/pii/S0045782522008064

## The repository includes: 
1. PINNs_TF: PINNs basic code using TensorFlow backend.
   The implementation solves the Poisson problem using a NN architecture
   with a collocation method for the loss function.

2. PINNs_jax: PINNs basic code using the JAX backend.
3. PINNs_pytorch: PINNs basic code using the TORCH backend.
       
4. DFR method in 1D  using TensorFlow backend.
   The implementation solves the Poisson problem using a NN architecture
   with a loss function based on the dual norm ($H^{-1}$) of the weak residual.
       
5. DFR method in 1D  using Jax backend.
  
6. DFR Method with hybrid optimizer based on Least-squares solver.
    - TODO: The vectorial derivatives need to be improved here and we are still missing the Jax version of it. 
  
       
