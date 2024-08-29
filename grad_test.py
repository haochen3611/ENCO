#%%
import torch
import torch.autograd as autograd

#%%

x = torch.randn(5,)

x.requires_grad_(True)

#%%

y = 0.5 *x ** 2
print(y)

# y.backward(torch.ones_like(y))


# %%

autograd.grad(y, x)

# %%
