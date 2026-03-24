# %%
import torch
import numpy as np

# Section 1: PyTorch
# (No code blocks in this section)

# Section 2: Tensors

# %%
# Subsection 2.1: Tensor initialization
data = [[1, 2], [3, 4]]         
x_data = torch.tensor(data)     # look just like x = np.array(data)

# %%
np_array = np.array(data)           # create a numpy array
x_np = torch.from_numpy(np_array)   # convert the numpy array to a tensor
print(x_np)

# %%
ones_tensor = torch.ones(size=(2, 3))       # Note the keyword argument "size", its counterpart in NumPy is "shape"
zeros_tensor = torch.zeros((2, 3))          # We don't have to use keyword argument though.
rand_tensor = torch.rand(size=(2, 3))

# You can also define the shape as tuple with the last empty element
shape = (2, 3,)
two_tensor = 2 * torch.ones(size=shape)     # the multiplication is element-wise operated

print(f"ones_tensor =\n{ones_tensor}\n")
print(f"zeros_tensor =\n{zeros_tensor}\n")
print(f"rand_tensor =\n{rand_tensor}\n")
print(f"two_tensor =\n{two_tensor}")

# %%
x = torch.tensor([[1, 2, 4], [4.4, 5.6, 6.5]], dtype=torch.int16)     # You can specify the data type too.
x_ones = torch.ones_like(x)

# if you don't define the dtype, it will give error as it will infer from 
# the data type of x as integer. However, is not possible for a number 
# between 0 and 1 to be an integer.
x_rand = torch.rand_like(x, dtype=torch.float32)   

print(f"x =\n{x}\n")            # the floating values are then round up to integers.
print(f"x_rand =\n{x_rand}\n")

# %%
# Subsection 2.2: Attributes of tensor
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
# My local machine (laptop) has Nvidia GPU. I also install pytorch on CUDA.
# Therefore, I can define a tensor on CUDA.
tensor = torch.rand(size=(3, 4), device='cuda')        
print(f"Device tensor is stored on: {tensor.device}")

# Section 3: Operations on Tensors

# %%
# Subsection 3.1: Standar numpy-like indexing and slicing
tensor = torch.ones((4, 4))
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

# %%
# Change all the values in the second column to 0
tensor[:,1] = 0
print(tensor)

# %%
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# %%
# Subsection 3.2: Arithmetic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# %%
print(f"a + b = {a.add(b)}")
print(f"a - b = {a.sub(b)}")
print(f"a * b = {a.mul(b)}")
print(f"a / b = {a.div(b)}")

# %%
a = torch.tensor([[1, 2, 3], [4, 5, 6]],  dtype=torch.float64)  # if you don't specify the data type, it will be inferred as integer.
b = torch.ones(size=(3, 4))
print(f"a @ b =\n{a.matmul(b)}")
print(f"a @ b =\n{a @ b}")

# %%
# Subsection 3.3: Aggregation
x = torch.ones(size=(2, 3, 4))
print(f"sum(x)  = {torch.sum(x)}\n")
print(f"x.sum() = {x.sum()}\n")

# We can perform aggregation along particular dimensions
print(f"x.sum(dim=2) =\n{x.sum(dim=2)}\n")
print(f"torch.sum(x, dim=2) =\n{torch.sum(x, dim=2)}\n")

# %%
x = torch.ones(size=(2, 3))
agg = x.sum()
agg_item = agg.item()
print(agg, "--", type(agg))
print(agg_item, "--", type(agg_item))

# %%
x = torch.ones(size=(2, 3))
x.add_(5)       # this is essentially equal to x += 5
print(x)

x = torch.ones(size=(2, 3))
x += 5
print(x)

# Subsection 3.4: Bridge with NumPy

# %%
# Subsection 3.5: Tensor to NumPy array
x_torch = torch.ones(5)     # this is a torch tensor

x_numpy = x_torch.numpy()         # convert a tensor to a numpy array
print(f"x_torch {type(x_torch)} =\n{repr(x_torch)}\n")  # repr is for developers, str is for normal users
print(f"x_numpy {type(x_numpy)} =\n{repr(x_numpy)}\n")

print("Difference between repr vs str\n" + 60*"=")
print(f"repr(x_numpy) = {repr(x_numpy)}")
print(f"str(x_numpy) = {str(x_numpy)} \n")

# You don't see the difference for tensor though
print(f"repr(x_torch) = {repr(x_torch)}")
print(f"str(x_torch) = {str(x_torch)}")

# %%
x_torch.add_(1)
print(f"x_torch =\n{repr(x_torch)}")
print(f"x_numpy =\n{repr(x_numpy)}")

# %%
# Subsection 3.6: NumPy array to Tensor
y_numpy = 2 * np.ones(5)
y_torch = torch.from_numpy(y_numpy)

# Again, a change the in NumPy array reflects in the tensor.
np.add(y_numpy, 1, out=y_numpy)    # this is just y_numpy += 1

print(f"y_torch = {y_torch}")
print(f"y_numpy = {y_numpy}")

# %%
x_numpy = np.random.rand(3, 2)
x_torch = torch.from_numpy(x_numpy)
print(f"x_numpy.dtype = {x_numpy.dtype}")
print(f"x_torch.dtype = {x_torch.dtype}")

# %%
x_torch = torch.rand(size=(3, 2))
x_numpy = x_torch.numpy()
print(f"x_numpy.dtype = {x_numpy.dtype}")
print(f"x_torch.dtype = {x_torch.dtype}")

# %%
x = torch.rand(size=(3, 2))
y = torch.from_numpy(np.ones(shape=(3, 2)))
print(x + y)    # it automatically converts to "bigger" datatype