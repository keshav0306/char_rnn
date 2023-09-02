import torch

a = torch.tensor([1,2,3], dtype=torch.float32, requires_grad=True)
d = a
a = a*a
b = a**2

print(b.data_ptr())
c = b
print(c.data_ptr())
b = torch.tensor(5, dtype=torch.float32, requires_grad=True)
print(b.data_ptr())
print(c.data_ptr())
a.retain_grad()
b.retain_grad()
c.sum().backward()
print(a.grad)
print(d.grad)
print(b.grad)