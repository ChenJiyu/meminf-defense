import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

class CIFAR10():
    def __init__(self, sigma=0.0):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x + sigma * torch.randn_like(x)])

    def load_dataset(self, batch_size=64):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        return trainloader, testloader
    
    def load_gen_dataset(self, batch_size=64, path='data/gen_data/'):
        data = np.array([]).reshape(0,3,32,32)
        labels = []
        import pickle
        for i in range(10):
            with open(path+str(i)+'.pkl','rb') as fp:
                nd = pickle.load(fp)
                data = np.concatenate((data, nd), axis=0)
                labels += [i for _ in range(nd.shape[0])]
        inds = [i for i in range(len(labels))]
        np.random.shuffle(inds)
        data = data[inds]
        labels = np.array(labels)[inds]
        dataloader = [(torch.FloatTensor(data[batch_size*i:batch_size*(i+1)]), \
                        torch.LongTensor(labels[batch_size*i:batch_size*(i+1)])) \
                        for i in range(int(data.shape[0]/batch_size))]
        return dataloader

class SVHN():
    def __init__(self, sigma=0.0):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x + sigma * torch.randn_like(x)])

    def load_dataset(self, batch_size=64):
        trainset = torchvision.datasets.SVHN(root='./data', transform=self.transform, download=True,
                                    split="train")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root='./data', transform=self.transform, download=True,
                                    split="test")
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        return trainloader, testloader
    
    def load_gen_dataset(self, batch_size=64, path='data/gen_data/'):
        data = np.array([]).reshape(0,3,32,32)
        labels = []
        import pickle
        for i in range(10):
            with open(path+str(i)+'.pkl','rb') as fp:
                nd = pickle.load(fp)
                data = np.concatenate((data, nd), axis=0)
                labels += [i for _ in range(nd.shape[0])]
        inds = [i for i in range(len(labels))]
        np.random.shuffle(inds)
        data = data[inds]
        labels = np.array(labels)[inds]
        dataloader = [(torch.FloatTensor(data[batch_size*i:batch_size*(i+1)]), \
                        torch.LongTensor(labels[batch_size*i:batch_size*(i+1)])) \
                        for i in range(int(data.shape[0]/batch_size))]
        return dataloader