# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original code from:
# https://github.com/wgrathwohl/JEM

# Useful libs
import os
import argparse
import random
import pickle
import numpy as np
import json
import sys
from tqdm import tqdm
from functools import reduce

# Pytorch libs
import torch as t, \
       torch.nn as nn, \
       torch.nn.functional as tnnF, \
       torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr

# Visualization libs
import matplotlib.pyplot as plt
from visdom import Visdom

# Miscellaneous model-related
from utils import utils
from utils import wideresnet
from utils.inception_score import inception_score
from utils.fid_score import calculate_fid_given_cifar
from utils.inception import InceptionV3

# Global vars
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 12
im_sz = 32
n_ch = 3

# defenses
from utils.meminf_defenses import DP_SGD, MIN_MAX
from utils.attack_model import MEMBERINF



class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, 
                                        dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, 
                                dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(args, bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, 
                    dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, \
                            "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
    
    f = f.to(device)
    return f, replay_buffer


class NoiseWrapper():
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x):
        return x + self.sigma * t.randn_like(x)


def get_data(args):
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
            #  lambda x: x + args.sigma * t.randn_like(x)]
             NoiseWrapper(args.sigma)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
            #  lambda x: x + args.sigma * t.randn_like(x)]
             NoiseWrapper(args.sigma)]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
        #  lambda x: x + args.sigma * t.randn_like(x)]
         NoiseWrapper(args.sigma)]
    )
    if not args.train_augment:
        transform_train = transform_test
    
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root,
                                transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root,
                                transform=transform, download=True, train=train)
        else:
            if train:
                if args.svhn_train_extra:
                    split_tag = 'extra'
                else:
                    split_tag = 'train'
            else:
                split_tag = 'test'
            return tv.datasets.SVHN(root=args.data_root, transform=transform,
                                                download=True, split=split_tag)

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))

    # set seed
    if args.fix_random_seed:
        np.random.seed(seed)
    
    # shuffle
    np.random.shuffle(all_inds)
    
    valid_inds, ref_inds, train_inds = (all_inds[:args.n_valid], 
                            all_inds[args.n_valid: args.n_valid + args.n_ref], 
                            all_inds[args.n_valid + args.n_ref:])
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(
                train_inds[train_labels == i][:args.labels_per_class]
                )
            other_inds.extend(
                train_inds[train_labels == i][args.labels_per_class:]
                )
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, 
                                    batch_size=args.batch_size, shuffle=True, 
                                    num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, 
                                                num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, 
                                                num_workers=4, drop_last=False)
    
    if args.min_max_defense:
        if args.dataset == 'cifar10':
            dset_ref = DataSubset(
                dataset_fn(True, transform_train),
                inds=ref_inds)
        else:
            dset_ex = tv.datasets.SVHN(root=args.data_root,
                        transform=transform_train, download=True, 
                        split='extra')
            dset_ref = DataSubset(
                dset_ex,
                inds=np.array(range(args.n_ref)))
        dload_ref = DataLoader(dset_ref, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, drop_last=True)
        dload_train_inf = DataLoader(dset_train, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=4, drop_last=True)
        return (dload_train, dload_ref, dload_train_inf, dload_train_labeled, 
                                                        dload_valid, dload_test)

    return dload_train, dload_train_labeled, dload_valid, dload_test

def get_data_retrain(args):
    print('loading retraining data from:', args.retrain_data_path)
    data = np.empty((0,3,32,32))
    labels = np.empty((0,))
    for i in range(10):
        p = os.path.join(args.retrain_data_path, str(i) + '.pkl')
        with open(p,'rb') as fp:
            data_i = pickle.load(fp)
            data = np.concatenate((data, data_i), axis=0)
            labels_i = i * np.ones(data_i.shape[0])
            labels = np.concatenate((labels, labels_i), axis=0)
    inds = list(range(len(labels)))
    np.random.shuffle(inds)
    data = data[inds]
    labels = labels[inds]

    data_train = data[:len(data) - args.n_ref]
    labels_train = labels[:len(data) - args.n_ref]

    dataloader = [(t.FloatTensor(
                        data_train[args.batch_size * i : args.batch_size * (i + 1)]
                        ),
                   t.LongTensor(
                        labels_train[args.batch_size * i : args.batch_size * (i + 1)])
                        )
                    for i in range(int(data_train.shape[0] / args.batch_size))]
    
    print('Size of re-train:', data_train.shape)
    if args.min_max_defense:
        data_ref = data[-args.n_ref:]
        labels_ref = labels[-args.n_ref:]
        inds = list(range(len(data_train)))
        np.random.shuffle(inds)
        data_train_inf = data_train[inds]
        labels_train_inf = labels_train[inds]
        dataloader_inf = [(t.FloatTensor(
                    data_train_inf[args.batch_size * i : args.batch_size * (i + 1)]
                    ),
                         t.LongTensor(
                    labels_train_inf[args.batch_size * i : args.batch_size * (i + 1)])
                    )
                    for i in range(int(data_train_inf.shape[0] / args.batch_size))]
        dataloader_ref = [(t.FloatTensor(
                    data_ref[args.batch_size * i : args.batch_size * (i + 1)]
                    ),
                         t.LongTensor(
                    labels_ref[args.batch_size * i : args.batch_size * (i + 1)])
                    )
                    for i in range(int(data_ref.shape[0] / args.batch_size))]
        print('Size of re-ref:', data_ref.shape)
        return dataloader, dataloader_inf, dataloader_ref

    return dataloader

def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        """
            Original sampling scheme: Fixed sized replay buffer,
            random pick entries, drop the old seeds in these entries. 
        """
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) \
                        if y is None else \
                      len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, \
                        "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()\
                                                        [:, None, None, None]
        samples = (choose_random * random_samples 
                    + (1 - choose_random) * buffer_samples)
        return samples.to(device), inds
    
    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """
            this func takes in replay_buffer now so we have the option to sample
            from scratch (i.e. replay_buffer==[]).
            See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples
        # (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(
                        f(x_k, y=y).sum(), [x_k], retain_graph=True
                        )[0]
            x_k.data += (args.sgld_lr * f_prime 
                        + args.sgld_std * t.randn_like(x_k))
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples

    return sample_q

def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss

def checkpoint(f, buffer, tag, args, device, f_inf=None):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer,
    }
    if args.min_max_defense:
        ckpt_dict["inf_state_dict"] = f_inf.state_dict()
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)



def main(args):

    # Save params, logs
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')


    # Fix seed to reproduce results
    if args.fix_random_seed:
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)


    # Whether load inception model for IS/FID
    if args.p_x_weight > 0 and args.compute_is_fid:
        from torchvision.models.inception import inception_v3
        import time
        print('Loading the godly slow inception V3...')
        ctime = time.time()
        inception_model_is = inception_v3(
                                pretrained=True, 
                                transform_input=False
                                ).type(t.cuda.FloatTensor)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model_fid = InceptionV3([block_idx])
        inception_model_fid.cuda()
        print('Loaded, took:', time.time() - ctime)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')


    # Datasets
    if args.min_max_defense:
        dload_train, dload_ref, dload_train_inf, \
        dload_train_labeled, dload_valid, dload_test = get_data(args)
        trainiter_inf = cycle(dload_train_inf)
        refiter_inf = cycle(dload_ref)
        t.save((dload_train_inf, dload_test), 
                os.path.join(args.save_dir, 'trainloader_inf.dl'))
    else:
        model_inf = None
        dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    if args.retrain:
        if args.min_max_defense:
            dataloader, dload_train_inf, dload_ref = get_data_retrain(args)
            dload_train_labeled = cycle(dataloader)
            trainiter_inf = cycle(dload_train_inf)
            refiter_inf = cycle(dload_ref)
            t.save((dload_train_inf, dload_test), 
                os.path.join(args.save_dir, 'trainloader_inf.dl'))
        else:
            dload_train_labeled = cycle(get_data_retrain(args))


    # Initialize replay buffer
    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)
    if args.new_buffer:
        replay_buffer = init_random(args, args.buffer_size)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(
                            t.clamp(x, -1, 1), p, 
                            normalize=True, nrow=sqrt(x.size(0))
                            )


    # Set up layer freezing
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    def flatten_f(name, f):
        """
            Return a list of all layers in the model
        """
        lst = []
        if len(list(f.named_children()))>0:
            for c in f.named_children():
                lst = lst + flatten_f(c[0],c[1])
        else:
            lst = [(name, f)]
        return lst
    # print(flatten_f('main_model',f))
    
    if args.freeze_conv_layers:
        layers = flatten_f('model', f)
        print('Total number of conv layers:', len(layers))
        for index, (name, layer) in enumerate(layers):
            if 'conv' in name and index < args.freeze_conv_num:
                print(layer)
                for p in layer.parameters():
                    if args.freeze_conv_fix:
                        p.requires_grad = False
            else:
                for p in layer.parameters():
                    # temp = t.zeros(p.shape).to(device)
                    temp = p.clone()
                    nn.init.normal_(p.data, mean=0.0, std=0.05)
                    p.data += temp


    # optimizer
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], 
                                                weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, 
                                                weight_decay=args.weight_decay)


    # Training stats recording
    best_valid_acc = 0.0
    best_test_acc = 0.0
    cur_iter = 0
    all_train_losses = [0]
    all_train_accs = [0]
    all_FIDs = [0]
    all_IS = [0]
    all_valid_acc = [0]
    IS = (0,0)
    FID = 0
    valid_acc = 0
    if args.send_visdom:
        viz = Visdom(
                port='Your port number', 
                server="http://localhost", 
                base_url='/', username='Your username', 
                password='Your password', 
                use_incoming_socket=True
                )
        win1 = viz.line(
                    X=np.array([0]), 
                    Y=np.column_stack((np.array([0]), 
                                       np.array([0]), 
                                       np.array([0]), 
                                       np.array([0]), 
                                       np.array([0]))), 
                    opts=dict(title=args.save_dir.split('/')[-1])
                    )
    

    # Defense initialization
    if args.dp:
        dp_def = DP_SGD(f, optim, device, 
                    dp_sigma=args.dp_sigma, l2_norm_clip=args.dp_l2_clip,
                    batch_size=args.batch_size, indiv=args.dp_single)
    if args.min_max_defense:
        mm_def = MIN_MAX(f, trainiter_inf, refiter_inf, device, lr=0.001,
                        momentum=0.9, weight_decay=5e-4, num_step=args.inf_step)
        model_inf = mm_def.model_inf 


    # Main training loops
    for epoch in range(args.start_epoch, args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.class_cond_p_x_sample:
                    assert not args.uncond, \
                        "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(
                            0, args.n_classes, (args.batch_size,)
                            ).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp

                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                l_p_x = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print(('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} '
                           'f(x_q)={:>14.9f} d={:>14.9f}').format(
                                epoch, i, fp, fq, fp - fq))
                L += args.p_x_weight * l_p_x
                if args.l2reg:
                    L += args.p_x_weight_l2reg * (l_p_x**2)

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab) if not args.dp_single else nn.CrossEntropyLoss(reduce=False)(logits, y_lab)

                # Min-max defense
                if args.min_max_defense:
                    l_p_y_given_x -= mm_def.update_loss(logits, l_p_y_given_x)

                acc = (logits.max(1)[1] == y_lab).float().mean()
                L += args.p_y_given_x_weight * l_p_y_given_x
                    
                if cur_iter % args.print_every == 0:
                    print(('P(y|x) {}:{:>d} loss={:>14.9f},'
                           ' acc={:>14.9f}').format(epoch,
                                            cur_iter,
                                            t.mean(l_p_y_given_x).item(),
                                            acc.item()))
                if args.l2reg:
                    L += args.p_y_given_x_weight_l2reg * (l_p_y_given_x**2)

            if args.p_x_y_weight > 0:  # maximize log p(x, y)
                assert not args.uncond, \
                    "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print(('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} '
                           'f(x_q)={:>14.9f} d={:>14.9f}').format(
                                epoch, i, fp, fq, fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if t.mean(L).abs().item() > 1e12:
                print("BAD BOIIIIIIIIII")
                1/0

            if args.dp is True: 
                dp_def.set_loss(L)
                dp_def.dp_sgd()
            else:
                optim.zero_grad()
                L.backward()

            optim.step()
            cur_iter += 1

            # Plot samples, save training stats
            if cur_iter % 100 == 1:
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, \
                            "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].\
                        repeat(args.n_classes, 1).transpose(1, 0).\
                        contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)
                
                if args.p_x_weight > 0 and args.compute_is_fid:
                    y = t.arange(0, args.n_classes)[None].\
                        repeat(50, 1).transpose(1, 0).\
                        contiguous().view(-1).to(device)
                    samples = sample_q(f, replay_buffer, y=y)
                    IS = inception_score(
                            inception_model_is,
                            t.clamp(samples, -1, 1).detach().cpu().numpy(),
                            resize=True
                            )
                    FID = calculate_fid_given_cifar(
                            inception_model_fid, 
                            t.clamp(samples, -1, 1).detach().cpu().numpy(),
                            batch_size=50,
                            dims=2048,
                            cuda=True
                            )
            all_train_accs.append(acc.item())
            all_train_losses.append(t.mean(L).item())
            all_IS.append(IS)
            all_FIDs.append(FID)
            all_valid_acc.append(valid_acc)

            if args.send_visdom:
                viz.line(X=np.array([cur_iter]), 
                     Y=np.column_stack((np.array([acc.item()]), 
                                        np.array([IS[0]/10]), 
                                        np.array([FID/100]), 
                                        np.array([L.item()/10]), 
                                        np.array([valid_acc]))), 
                     win=win1, update='append')

        # Saving training stats
        with open(args.save_dir + '/' +'training_stats.pkl', 'wb') as fp:
            pickle.dump((all_train_accs, all_train_losses, 
                         all_valid_acc, all_IS, all_FIDs), fp)

        # Saving models
        if epoch % args.ckpt_every == 0:
                checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', 
                           args, device, f_inf=model_inf)

        if epoch % args.eval_every == 0 \
                    and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", 
                               args, device, f_inf=model_inf)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
                if correct > best_test_acc:
                    best_test_acc = correct
                    print("Best Test!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_test_ckpt.pt", 
                               args, device, f_inf=model_inf)
                valid_acc = correct
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", 
                    args, device, f_inf=model_inf)

def sample_from_model(args):
    """
        Sample a dataset from a trained JEM. Use with args.sample_mode.
    """
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Load replay buffer
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, \
                            "Buffer size must be divisible by args.n_classes"
    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)
    if args.new_buffer:
        replay_buffer = init_random(args, args.buffer_size)

    #original model
    model_cls_ori = F if args.uncond else CCF
    f_ori = model_cls_ori(args.depth, args.width, args.norm, 
                    dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    
    if args.dataset == 'cifar10':
        print(f"loading pre-trained model...")
        ckpt_dict = t.load('YOUR_PRE_TRAINED_CLASSIFIER')
    elif args.dataset == 'svhn':
        print(f"loading pre-trained model...")
        ckpt_dict = t.load('YOUR_PRE_TRAINED_CLASSIFIER')
    f_ori.load_state_dict(ckpt_dict["model_state_dict"])
    f_ori = f_ori.to(device)

    num_ea_class = 50
    logi_thres = args.sample_thres
    save_size = args.sample_size
    
    sample_per_class = [np.array([]).reshape(0,3,32,32) for _ in range(args.n_classes)]

    save_flag = [False for i in range(args.n_classes)]

    sample_save_dir = ('retrain_samples_' + args.load_path.split('/')[-2] 
                        + '_' + args.load_path.split('/')[-1] \
                        + '_step_' + str(args.n_steps) \
                        + '_scratch_' + str(args.new_buffer) \
                        + '_buffsize_' + str(args.buffer_size) + '/')
    if not os.path.exists(sample_save_dir):
        os.mkdir(sample_save_dir)
    
    for _ in tqdm(range(10000)):
        if args.new_buffer:
            replay_buffer = init_random(args, args.buffer_size)

        if False not in save_flag:
            break
        y = t.arange(0, args.n_classes)[None].\
            repeat(num_ea_class, 1).\
            transpose(1, 0).contiguous().view(-1).to(device)
        y = y.detach().cpu().numpy()
        for l in range(len(save_flag)):
            if save_flag[l]:
                y = y[y!=l]
        y = t.LongTensor(y).to(device)
        x_q_y = sample_q(f, replay_buffer, y=y)
        
        out_ori = f_ori.classify(x_q_y)
        logits_ori = t.max(t.softmax(out_ori, dim=1),dim=1)[0].\
                                                        detach().cpu().numpy()
        prediction_ori = t.max(t.softmax(out_ori, dim=1),dim=1)[1].\
                                                        detach().cpu().numpy()
        filter_ori = ((y.detach().cpu().numpy() == prediction_ori) 
                      & (logits_ori > logi_thres))

        val_sample = x_q_y.detach().cpu().numpy()[filter_ori]
        val_label = y.detach().cpu().numpy()[filter_ori]
        for i in range(args.n_classes):
            if save_flag[i]:
                continue
            sample_per_class[i] = np.concatenate((sample_per_class[i], 
                                            val_sample[val_label==i]), axis=0)
            print('class:', i, sample_per_class[i].shape)
            if sample_per_class[i].shape[0] >= save_size:
                with open(os.path.join(sample_save_dir, str(i)+'.pkl'), 'wb') as fp:
                    pickle.dump(sample_per_class[i][:save_size], fp)
                    save_flag[i] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit", fromfile_prefix_chars='@')
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", 
                        help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    parser.add_argument("--freeze_conv_layers", action="store_true", 
                        help="If set, freeze some conv layers")
    parser.add_argument("--freeze_conv_fix", action="store_true", 
                        help="If true, grad=False for frozen conv layers")
    parser.add_argument("--freeze_conv_num", type=int, default=1e9, 
                        help="Number of conv layers to freeze")

    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_weight_l2reg", type=float, default=0.01)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight_l2reg", type=float, default=0.01)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)

    # regularization
    parser.add_argument("--l2reg", action="store_true", 
                        help="If true, add l2 reg to loss")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")

    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, 
                        help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, 
                        help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", 
                        help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--new_buffer", action="store_true", 
                        help="Initialize as random buffer")

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10,
                        help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, 
                        help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, 
                        help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", 
                        help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", 
                        help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", 
                        help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--send_visdom", action="store_true", 
                        help="Send training stats to visdom")
    parser.add_argument("--compute_is_fid", action="store_true", 
                        help="Compute IS/FID")

    # Other training options
    parser.add_argument("--fix_random_seed", action="store_true", 
                        help="If true, use a fixed random seed to make results reproducable")
    parser.add_argument("--train_augment", action="store_true", 
                        help="If true, apply data augmentation on the training dataset")
    parser.add_argument("--retrain", action="store_true", 
                        help="If true, retrain with customized dataset")
    parser.add_argument("--retrain_data_path", type=str, default=None)
    parser.add_argument("--sample_mode", action="store_true", 
                        help="If true, generate samples from loaded model")
    parser.add_argument("--sample_size", type=int, default=6000, 
                        help="sample size for each class label in sample_mode")
    parser.add_argument("--sample_thres", type=float, default=0.5, 
                        help="threshold of sample selection")
    parser.add_argument("--svhn_train_extra", action="store_true", 
                        help="When dataset=svhn, use extra set instead of train set")
    
    # Min-max defense
    parser.add_argument("--min_max_defense", action="store_true", 
                        help="If set, train with min-max defense")
    parser.add_argument("--n_ref", type=int, default=0)
    parser.add_argument("--inf_step", type=int, default=5)

    # Differential Privacy
    parser.add_argument("--dp", action="store_true", 
                        help="If true, train with differential privacy")
    parser.add_argument("--dp_single", action="store_true", 
                        help="If true, compute gradient for each single sample in a batch")
    parser.add_argument("--dp_l2_clip", type=float, default=1.0, help="l2 norm clip for differential privacy")
    parser.add_argument("--dp_sigma", type=float, default=1.9, help="controls noise multiplier for differential privacy")


    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    if args.sample_mode:
        sample_from_model(args)
    else:
        main(args)
