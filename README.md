# Linear and nonlinear disambiguation optimization (LANDO)
This repository contains the codes used to perform the LANDO algorithm.
The framework is described in the following schematic and reference

<br/>
<p align="center"> 
<img src="images/schematic.png?raw=true" width="700px">
</p>
<br/>

The code allows you to generate all the data used in the paper. Alternatively, you can download the data directly [at this Dropbox link](https://www.dropbox.com/sh/ym5iypkjontzydy/AACnYWmXGoIZt7VXSl-DM0waa?dl=0).

If you want to generate the data manually then you need to install [Chebfun](www.chebfun.org).

Here's a brief summary of the main codes. Further details can be found in the function files, or by calling, for example, "help trainLANDO".

File | Description
------------ | -------------
```defineKernel.m``` | Defines a kernel cell based on provided hyperparameters
```trainLANDO.m``` | Trains a LANDO model based on data, sparsification parameter and kernel
```linopLANDO.m```| Extracts and analyses the linear component of the LANDO model relative to a given base state
```predictLANDO.m``` | Forms predictions based on a given LANDO model


To get started, add the source and example folders to the path (```addpath('src','examples')```) and  try running ```lorenzExample.m```

Please contact me if you encounter any bugs or have requests for features.

## Reference:
_Kernel Learning for Robust Dynamic Mode Decomposition: Linear and Nonlinear Disambiguation Optimization (LANDO)_  
Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon & Steven L. Brunton  
[arXiv:2106.01510](https://arxiv.org/abs/2106.01510)
