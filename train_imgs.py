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
    mgrid = mgrid.reshape(sidelengths[0], -1, len(sidelengths))
    return mgrid

# Used for generating indexes. Currently unused
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
    def __init__(self, sidelength, img_list=[], debug=False):
        super().__init__()

        img_set = set(img_list)

        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            #Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
        ])

        imgs = []
        for (root,_,files) in os.walk('data/48/'):
            for i, name in enumerate(files):
                if i not in img_set:
                    continue

                filepath = os.path.join(root, name)
                with Image.open(filepath).convert('RGB') as img:
                    if debug:
                        print(f'{i}: {filepath}')
                    img = transform(img)
                    img = img.permute(1,2,0).view(-1,3)
                    imgs.append(img)

        self.imgs = imgs
        self.length = len(imgs)

        # Idxs to pull out the correct image pixel
        #self.idxs = get_rgrid([len(imgs), sidelength, sidelength])
        # Coordinates to pass to NN
        self.coords = get_mgrid([self.length, sidelength, sidelength])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.imgs[idx]

    def input_grid(self, width, num_mult=1):
        '''
        Create an input grid for testing

        Args:
            num_mult: how much to muliply the number of images by
        '''

        return get_mgrid([self.length * num_mult, width, width])


def gpu_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'Total:{t} Reserved:{r} Allocated:{a}')

def run(args):
    sidelength = 48
    dir_name = f'{sidelength}'

    if args.img_list == []:
        args.img_list = range(args.num_images)
        dir_name += f'_n{args.num_images}'
    else:
        args.num_images = len(args.img_list)
        dir_name += f'_l{args.img_list[0]}-{args.img_list[1]}'

    if args.upsample:
        dir_name += f'_up{args.upsample}'

    if args.interpolate:
        dir_name += '_i'

    out_path = os.path.join('.', 'imgs', dir_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = ImageFitting(sidelength, img_list=args.img_list, debug=args.create_list)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=3, out_features=3, hidden_features=200,
                    hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    epochs = 500
    epochs_til_summary = 100

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    test_length = args.upsample if args.upsample else sidelength
    num_mult = 5 if args.interpolate else 1
    test_input = dataset.input_grid(test_length, num_mult=num_mult).cuda()

    for epoch in range(1, args.epochs + 1):
        all_img_loss = 0.
        for imgnum, (model_input, ground_truth) in enumerate(dataloader):

            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            model_output, coords = img_siren(model_input)
            loss = ((model_output - ground_truth)**2).mean()
            all_img_loss += loss

            #gpu_info()

            def show_image(idx):
                if not args.save_image and not args.show_image:
                    return

                with torch.no_grad():
                    test_output = img_siren(test_input[idx])[0] # get test image
                output = test_output.cpu().view(test_length, test_length,3).numpy()
                np.clip(output, 0., 1., output)
                plt.imshow(output)
                if args.show_image:
                    plt.show()
                if args.save_image:
                    path = os.path.join(out_path, f'e{epoch:04d}_{idx:03d}.png')
                    plt.savefig(path)
                plt.close()

            if not (epoch % epochs_til_summary):
                if not args.interpolate:
                    show_image(imgnum)
                else:
                    for i in range(num_mult):
                        show_image(imgnum * num_mult + i)

            optim.zero_grad()
            loss.backward()
            optim.step()

        all_img_loss /= args.num_images
        print(f"Epoch {epoch} L2 loss over all images {all_img_loss:0.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Sine activation networks')
    parser.add_argument('--num-images', help='How many images to learn', type=int, default=100)
    parser.add_argument('--show-image', help='Show imgs on-screen', default=False, action='store_true')
    parser.add_argument('--no-save-image', dest='save_image', help='Save imgs to disk', default=True, action='store_false')
    parser.add_argument('--epochs', help='How long to go for', type=int, default=1000)
    parser.add_argument('--upsample', help='Upsample to a higher dimension', type=int, default=None)
    parser.add_argument('--create-list', help='Map number to file name', default=False, action='store_true')
    parser.add_argument('--img-list', help='Choose 2 images for interpolation', nargs=2, default=[])
    parser.add_argument('--interpolate', help='Interpolate between images', default=False, action='store_true')

    args = parser.parse_args()

    args.img_list = [int(x) for x in args.img_list]

    run(args)

