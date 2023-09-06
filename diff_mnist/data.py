import os, shutil
import urllib
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image
import math
import copy
from scipy.interpolate import griddata


# build dataset class out of the above changes to MNIST
class MNIST_Triangular(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False, to_square_preprocess=False):
        self.data = torchvision.datasets.MNIST(root=root, train=train, download=download).data  # ugly but works
        # calculate background color
        background_color = self.data[:, 0:2, 0:2].numpy().mean()
        # print("background_color: ", background_color)

        # print(train_dataset.data.shape)

        # normalise data between 0 and 1
        self.data = self.data.float() / 255.0

        # new dataset
        new_data = torch.zeros((self.data.shape[0], 64, 64))
        # make all pixels not assigned otherwise gray (background_color)
        new_data += background_color
        # place the data in the bottom right corner
        shift = 5

        # new_data[:, 0+shift:28+shift, 0+shift:28+shift] = self.data.float()  # in top right
        new_data[:, -(0+shift+28):-shift, 0+shift:28+shift] = self.data.float()  # in top right

        # make the lower-right half of the image gray
        # this means everything below the diagonal is gray
        gray = .5
        for i in range(64):
            # color bottom right
            # new_data[:, i, 64-i:] = gray  # gray OR background_color
            # color top right
            new_data[:, i, i:] = gray  # gray OR background_color

        if to_square_preprocess: 
            # only take subset because it takes a lot of time
            # new_data = new_data[:500]

            print("preprocessing to square")
            # to numpy 
            device = new_data.device
            new_data = new_data.cpu().numpy()

            print("creating class")
            preprocessor = Preprocess_triangular(J=int(math.log2(new_data.shape[1])))
            square_img_list = []
            print("starting the loop")
            for i, img in enumerate(new_data): 
                if i % 1000 == 0 or i in [1, 2]:
                    print("processing image to square: ", i, " of ", len(new_data))

                square_img = preprocessor.process_mnist_triangular(img)
                square_img_list.append(square_img)
            
            # back to torch
            new_data = torch.tensor(square_img_list).to(torch.float32).to(device)  # float()


        # insert channel dim
        new_data = new_data.unsqueeze(1)

        # downsample data to 32x32
        # new_data = torch.nn.functional.avg_pool2d(new_data, 2, 2)

        # plotting for debugging
        # import matplotlib.pyplot as plt
        # plt.imshow(new_data[0, 0])

        self.data = new_data
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Preprocess_triangular(): 
    def __init__(self, J) -> None:
        self.J = J

        f1 = lambda x : [x[0]/2 , x[1]/2]
        f2 = lambda x : [x[0]/2 , x[1]/2 + 0.5]
        f3 = lambda x : [x[0]/2 + 0.5 , x[1]/2]
        f4 = lambda x : [x[0]/2 + 0.5 , x[1]/2 + 0.5]

        F_square = [f1,f2,f3,f4]

        f1 = lambda x : [x[0]/2 , x[1]/2]
        f2 = lambda x : [x[0]/2 , x[1]/2 + 0.5]
        f3 = lambda x : [x[0]/2 + 0.5 , x[1]/2]
        f4 = lambda x : [-x[0]/2 + 0.5 , -x[1]/2 + 0.5]

        F_tri = [f1,f2,f3,f4]

        addresses = get_addresses(J=J)  # tensor.shape[1]

        self.square_array = np.array(get_eval_points(F_square, addresses, x_center = [0.5,0.5]))
        self.tri_array = np.array(get_eval_points(F_tri, addresses, x_center = [1/3,1/3]))

        print("class init done")


    def process_mnist_triangular(self, tensor):
        
        # tensor = tensor.cpu()
        # tensor = tensor.squeeze()
        image = np.rot90(tensor, 3)  # tensor.numpy()
        
        img = swap_array(image, self.square_array, self.tri_array).reshape(2**(self.J), 2**(self.J))
        
        # img = torch.from_numpy(img).unsqueeze(0)
        
        return img


def get_eval_points(F, addresses,x_center = [0,0]):
    
    eval_points = copy.deepcopy(addresses)

    n = len(addresses)

    for i in range(n):
        for j in range(n):

            string = addresses[i][j]
            
            x = x_center 

            for k in reversed(string):
                
                x = F[int(k)](x)

            eval_points[i][j] = x
    
    return eval_points

def swap_array(img,in_array,out_array,method='nearest'):
    
    m = out_array.shape[0]
    
    xyz_data = np.dstack((in_array,img))

    xyz_arr = xyz_data.reshape(-1,3)

    target_points = out_array.reshape(-1,2)

    values = griddata(xyz_arr[:,:2], xyz_arr[:,2], target_points, method=method)

    values = values.reshape(m,m,-1)
    
    return values


def string_kronecker_product(matrix1, matrix2):
    # Get the dimensions of the matrices
    n1, m1 = len(matrix1), len(matrix1[0])
    n2, m2 = len(matrix2), len(matrix2[0])

    # Initialize the result matrix
    result = [['' for _ in range(m1*m2)] for _ in range(n1*n2)]

    # Compute the Kronecker product
    for i in range(n1):
        for j in range(m1):
            for k in range(n2):
                for l in range(m2):
                    result[n2*i + k][m2*j + l] = matrix1[i][j] + matrix2[k][l]
    return result

def get_addresses(J):

    # Test the function
    matrix = [['0', '1'], ['2', '3']]

    addresses = matrix

    for idx, _ in enumerate(range(J-1)):
        # print("get_addresses: ", idx)
        addresses = string_kronecker_product(addresses, matrix)

    return addresses




def swap_array(img,in_array,out_array,method='nearest'):
    
    m = out_array.shape[0]
    
    xyz_data = np.dstack((in_array,img))

    xyz_arr = xyz_data.reshape(-1,3)

    target_points = out_array.reshape(-1,2)

    values = griddata(xyz_arr[:,:2], xyz_arr[:,2], target_points, method=method)

    values = values.reshape(m,m,-1)
    
    return values






class MNIST(Dataset):
    def __init__(
        self,
        root="./dataset",
        load=True,
        source_root=None,
        imageSize=28,
        train=True,
        num_channels=1,
        device="cpu",
    ):  # load=True means loading the dataset from existed files.
        super(MNIST, self).__init__()
        self.num_channels = min(3, num_channels)
        if load:
            self.data = torch.load(os.path.join(root, "data.pt"))
            self.targets = torch.load(os.path.join(root, "targets.pt"))
        else:
            if source_root is None:
                source_root = "./datasets"

            source_data = torchvision.datasets.MNIST(
                source_root,
                train=train,
                transform=transforms.Compose(
                    [
                        transforms.Resize(imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
                download=True,
            )
            self.data = torch.zeros((0, self.num_channels, imageSize, imageSize))
            self.targets = torch.zeros((0), dtype=torch.int64)
            # has 60000 images in total
            dataloader_R = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_G = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_B = DataLoader(source_data, batch_size=100, shuffle=True)

            im_dir = root + "/im"
            if os.path.exists(im_dir):
                shutil.rmtree(im_dir)
            os.makedirs(im_dir)

            idx = 0
            for (xR, yR), (xG, yG), (xB, yB) in tqdm(
                zip(dataloader_R, dataloader_G, dataloader_B)
            ):
                x = torch.cat([xR, xG, xB][-self.num_channels :], dim=1)
                y = (100 * yR + 10 * yG + yB) % 10**self.num_channels
                self.data = torch.cat((self.data, x), dim=0)
                self.targets = torch.cat((self.targets, y), dim=0)

                for k in range(100):
                    if idx < 10000:
                        im = x[k]
                        filename = root + "/im/{:05}.jpg".format(idx)
                        save_image(im, filename)
                    idx += 1

            if not os.path.isdir(root):
                os.makedirs(root)
            torch.save(self.data, os.path.join(root, "data.pt"))
            torch.save(self.targets, os.path.join(root, "targets.pt"))
            vutils.save_image(x, "ali.png", nrow=10)

        self.data = self.data.to(device)
        self.targets = self.targets.to(device)

        # Pad to make 32 x 32 so that resolution can be repeatedly halved
        self.data = torch.nn.functional.pad(self.data, pad=(2, 2, 2, 2))


    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        return img, targets

    def __len__(self):
        return len(self.targets)



# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def get_celeba_datasets(): 
    resize = 64
    # num_classes = 40
    train_transform, valid_transform = _data_transforms_celeba64(resize)
    train_data = LMDBDataset(root='datasets/celeba64_lmdb', name='celeba64', split="train", transform=train_transform, is_encoded=True)
    valid_data = LMDBDataset(root='datasets/celeba64_lmdb', name='celeba64', split="validation", transform=valid_transform, is_encoded=True)
    test_data = LMDBDataset(root='datasets/celeba64_lmdb', name='celeba64', split="test", transform=valid_transform, is_encoded=True)

    X_train = np.zeros((len(train_data), resize, resize, 3))
    for i in range(len(train_data)):
        X_train[i,:,:,:] = np.einsum('ijk -> jki', train_data[i])

    X_val = np.zeros((len(valid_data), resize, resize, 3))
    for i in range(len(valid_data)):
        X_val[i,:,:,:] = np.einsum('ijk -> jki', valid_data[i])

    X_test = np.zeros((len(test_data), resize, resize, 3))
    for i in range(len(test_data)):
        X_test[i,:,:,:] = np.einsum('ijk -> jki', test_data[i])

    # ------ treating as int (START)

    # scale to [0, 255] as expected
    # X_train = X_train * 255.
    # X_val = X_val * 255.
    # X_test = X_test * 255.

    # round to nearest integer
    # X_train = np.round(X_train, decimals=0)
    # X_val = np.round(X_val, decimals=0)
    # X_test = np.round(X_test, decimals=0)

    # convert to uint8
    # X_train = X_train.astype('uint8')
    # X_val = X_val.astype('uint8')
    # X_test = X_test.astype('uint8')

    # ------ treating as int (END)

    # convert to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    # standardize the data to have zero mean and unit variance, computed on the training set
    # norm_mean = np.mean(X_train, axis=(0, 1, 2, 3))
    # norm_std = np.std(X_train, axis=(0, 1, 2, 3))
    # X_train = (X_train - norm_mean) / (norm_std + 1e-7)
    # X_val = (X_val - norm_mean) / (norm_std + 1e-7)
    # X_test = (X_test - norm_mean) / (norm_std + 1e-7)
    norm_mean, norm_std = None, None

    # convert to torch tensor
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)

    # move channel dim from 3rd to 1st dim
    X_train = X_train.permute(0, 3, 1, 2)
    X_val = X_val.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    # currently in [0, 1], now normalize st between [-1, 1]
    # X_train = X_train * 2 - 1
    # X_val = X_val * 2 - 1
    # X_test = X_test * 2 - 1

    # TODO produced error with generator --> remove once other works
    # dataset_train = torch.utils.data.TensorDataset(torch.as_tensor(X_train, dtype=torch.float32))
    # dataset_val = torch.utils.data.TensorDataset(torch.as_tensor(X_val, dtype=torch.float32))
    # dataset_test = torch.utils.data.TensorDataset(torch.as_tensor(X_test, dtype=torch.float32)) 

    # TODO JUST FOR TESTING PURPOSES
    # X_train = torch.zeros((1000, 3, 64, 64))
    # X_val = torch.zeros((1000, 3, 64, 64))
    # X_test = torch.zeros((1000, 3, 64, 64))
    # norm_mean = 0
    # norm_std = 1

    dataset_train = SimpleDataset(X_train)
    dataset_val = SimpleDataset(X_val)
    dataset_test = SimpleDataset(X_test)

    return dataset_train, dataset_val, dataset_test, norm_mean, norm_std


def unnormalize_fn(x, mean, std):
    return x * (std + 1e-7) + mean


class SimpleDataset(torch.utils.data.TensorDataset): 
    """
    Without this dataset, and just using a TensorDataset doesn't work with the generator, because it calls the __getitem__ method of the Dataset base class
    which is misdefined for this data, and then thrwos an error.
    """
    def __init__(self, data): 
        super(SimpleDataset, self).__init__(data) 
        self.data = data

    def __getitem__(self, index):
        img = self.data[index]

        return img

    def __len__(self):
        return len(self.data)



def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),  # taken out compared to NVAE --> we don't want data augmentation
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def num_samples(dataset, split):
    if dataset == 'celeba':
        # return 27000 if train else 3000
        pass
    elif dataset == 'celeba64':
        if split == "train":
            return 162770
        elif split == "validation":
            return 19867
        elif split == "test":
            return 19962
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', split="train", transform=None, is_encoded=False):
        self.name = name
        self.split = split
        self.transform = transform
        if self.split in ["train", "validation", "test"]:
            lmdb_path = os.path.join(root, f'{self.split}.lmdb')
        else:
            print('invalid split param')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return num_samples(self.name, self.split)