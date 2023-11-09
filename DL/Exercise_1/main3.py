import torch
import torch.nn as nn

## data
x = torch.Tensor([0.8345,0.0993,1.8054,1.8896,0.9817])
y = torch.Tensor([0.9785,0.6754,1.8001,0.7385,0.2224])

value = torch.Tensor([8.4596,2.2981,13.8385,11.3696,4.7279])


## model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        self.c = nn.Parameter(torch.rand(1))


    def forward(self, x, y):

        f = self.a * x + self.b * y + self.c
        return f

## training scheme
model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for i in range(1000):
    f = model(x, y)
    e = torch.abs(f - value)
    cost = e.mean()

    if i % 100 == 1:
        print('a: %f, b: %f, c: %f, cost: %f' % (model.a.data, model.b.data, model.c.data, cost.data))

    # clear old gradients
    optimizer.zero_grad()
    cost.backward()

    optimizer.step()

print('done')


