import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input = torch.randint(high=255, size=(1, 3, 10, 10))
f = torch.fft.fft2(input)
fshift = torch.fft.fftshift(f)
magnitude = 20*torch.log(torch.abs(fshift))
print(magnitude.shape)

# Visualization Functions
def tensor_to_numpy(tensor):
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img

plt.imshow(tensor_to_numpy(torch.squeeze(input, dim=0)))
plt.show()

magnitude_image = tensor_to_numpy(torch.squeeze(magnitude, dim=0))
plt.imshow(magnitude_image)
plt.show()