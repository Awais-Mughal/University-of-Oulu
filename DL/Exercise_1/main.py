
import torch
import torch.nn

## data

x = torch.Tensor([0.8345,0.0993,1.8054,1.8896,0.9817])
y = torch.Tensor([0.9785,0.6754,1.8001,0.7385,0.2224])

value = torch.Tensor([8.4596,2.2981,13.8385,11.3696,4.7279])

## model

a = torch.rand(1)
b = torch.rand(1)
c = torch.rand(1)

a.requires_grad = True
b.requires_grad = True
c.requires_grad = True

def forward(x, y, a, b, c):
    f = a * x + b * y + c # is a vector , 1x5
    e = torch.abs(f -  value) # e is a vector, 1x5
    cost = e.mean() # is a scalar
    return cost

## training scheme
lr = 0.1

for i in range(1000):

    cost = forward(x, y, a, b, c)

    if i % 100 == 1:
        print('a: %f, b: %f, c: %f, cost: %f' % (a.data, b.data, c.data, cost.data))

    # calculate the gradients of a, b, c
    a.grad = None
    b.grad = None
    c.grad = None
    cost.backward()

    # update a, b, c with gradients
    a.data = a.data - lr * a.grad
    b.data = b.data - lr * b.grad
    c.data = c.data - lr * c.grad

print('done')
