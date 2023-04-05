import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor
from mindspore import nn


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    x = ops.expand_dims(x, 1)
    y = ops.expand_dims(y, 0)
    new_size = Tensor(np.array([x_size, y_size, dim]), dtype=mindspore.int32)
    titled_x = ops.expand(x, new_size)
    titled_y = ops.expand(y, new_size)
    kernel_input = (titled_x - titled_y).pow(2).mean(2) / float(dim)
    return ops.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def compute_mmd_loss(batch):
    z = batch["z"]
    true_samples = ops.standard_normal(z.shape)
    true_samples = ops.stop_gradient(true_samples)
    loss = compute_mmd(true_samples, z)
    return loss

def compute_kl_loss(batch):
    mu, logvar = batch["mu"], batch["logvar"]
    temp = 1 + logvar - mu.pow(2) - logvar.exp()
    loss = -0.5 * temp.sum()
    return loss

def compute_rc_loss(batch):
    x = batch["x"]
    output = batch["output"]
    gtmsk = ops.permute(x, (0, 3, 1, 2))
    outmsk = ops.permute(output, (0, 3, 1, 2))
    lossfun = nn.MSELoss()
    loss = lossfun(gtmsk, outmsk)
    return loss

def compute_rcxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]

    gtmasked = ops.permute(x, (0, 3, 1, 2))[mask]
    outmasked = ops.permute(output, (0, 3, 1, 2))[mask]
    
    lossfun = nn.MSELoss()
    loss = lossfun(gtmsk, outmsk)
    return loss

_matching_ = {"rc": compute_rc_loss, "kl": compute_kl_loss, "rcxyz": compute_rcxyz_loss}

def get_loss_function(ltype):
    return _matching_[ltype]

def get_loss_names():
    return list(_matching_.keys())
