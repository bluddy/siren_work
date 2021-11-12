import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

from sine import Siren

def get_mgrid(sidelengths):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors =  [torch.linspace(-1, 1, steps=s) for s in sidelengths]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(100, -1, len(sidelengths))
    return mgrid

def get_rgrid(sidelengths):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors =  [torch.LongTensor(range(s)) for s in sidelengths]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelengths))
    return mgrid

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()

        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            #Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
        ])

        imgs = []
        for (root,_,files) in os.walk('data/48/'):
            for name in files:
                with Image.open(os.path.join(root, name)).convert('RGB') as img:
                    img = transform(img)
                    img = img.permute(1,2,0).view(-1,3)
                    imgs.append(img)
        self.imgs = imgs
        # Idxs to pull out the correct image pixel
        #self.idxs = get_rgrid([len(imgs), sidelength, sidelength])
        # Coordinates to pass to NN
        self.coords = get_mgrid([len(imgs), sidelength, sidelength])
        self.length = len(imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.imgs[idx]

def gpu_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'Total:{t} Reserved:{r} Allocated:{a}')

def run():
    sidelength = 48
    dataset = ImageFitting(sidelength)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=3, out_features=3, hidden_features=200,
                    hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    epochs = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    epochs_til_summary = 100

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    for epoch in range(epochs):
        # Iterate over images
        for imgnum, (model_input, ground_truth) in enumerate(dataloader):
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            model_output, coords = img_siren(model_input)
            loss = ((model_output - ground_truth)**2).mean()

            #gpu_info()

            def show_imgs():
                img_grad = gradient(model_output, coords)
                img_laplacian = laplace(model_output, coords)

                fig, axes = plt.subplots(1,3, figsize=(18,6))
                axes[0].imshow(model_output.cpu().view(sidelength,sidelength,3).detach().numpy())
                axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy())
                axes[2].imshow(img_laplacian.cpu().view(sidelength,sidelength).detach().numpy())
                plt.show()

            print(f"Epoch {epoch} imgnum {imgnum}, Total loss {loss:0.6f}")

            if not (epoch % epochs_til_summary) and imgnum == 0:
                show_imgs()

            if not imgnum and epoch == epochs - epochs_til_summary:
                show_imgs()

            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == '__main__':
    run()
