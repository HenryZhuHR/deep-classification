import numpy as np
import torch

input = torch.ones([1,2], requires_grad=False)

w1 = torch.tensor(np.array([[2],[2]],dtype=np.float32), requires_grad=True)
b1=torch.tensor(1., requires_grad=True)

w2 = torch.tensor(np.array([[3],[3]],dtype=np.float32), requires_grad=True)
b2 = torch.tensor(2., requires_grad=True)


# compute
h1=torch.mm(input,w1)+b1
h2=torch.mm(input,w2)+b2
h3=h1*h2
loss=h3/2

h1.retain_grad()
h2.retain_grad()
h3.retain_grad()
loss.retain_grad()


print('h1 =',h1)
print('h2 =',h2)
print('h3 =',h3)
print('loss =',loss)
print()


loss.backward()
print('h1 grad =',h1.grad)
print('h2 grad =',h2.grad)
print('h3 grad =',h3.grad)
print('loss grad =',loss.grad)
print()