import os
import numpy as np
import random
import pickle
from tqdm import tqdm
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import CIFAR10, SVHN
from train_jem import *

def gen_meminf(load_from, save_attack_data=None, overwrite=False):
    if not os.path.exists('meminf_data'):
        os.mkdir('meminf_data')

    if save_attack_data is None:
        if 'cifar' in load_from:
            save_attack_data = os.path.join('meminf_data/cifar10', load_from.split('/')[-2] + '.pkl')
        else:
            save_attack_data = os.path.join('meminf_data/svhn', load_from.split('/')[-2] + '.pkl')
    else:
        save_attack_data = save_attack_data
    
    print('Load from:', load_from)
    if os.path.exists(save_attack_data) and not overwrite:
        print('Exist:', save_attack_data)
        return
    print('Save to:', save_attack_data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader_dir = os.path.join('/'.join(load_from.split('/')[:-1]), 'trainloader_inf.dl')
    if os.path.exists(trainloader_dir) and 'retrain' not in load_from:
        trainloader, testloader = torch.load(trainloader_dir)
    else:
        if 'cifar' in load_from:
            trainloader, testloader = CIFAR10(sigma=0.0).load_dataset()
        elif 'svhn' in load_from:
            trainloader, testloader = SVHN(sigma=0.0).load_dataset()

    net = CCF(28, 10, None, dropout_rate=0.0, n_classes=10)
    ckpt_dict = t.load(load_from)
    net.load_state_dict(ckpt_dict["model_state_dict"])
    net = net.to(device)

    #save input and output for train and test data (true_label, pred_logits, in_or_out)
    all_labels_test = []
    all_outputs_test = []
    all_labels_train = []
    all_outputs_train = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net.classify(images)
            all_labels_test.append(labels.detach().cpu().numpy())
            all_outputs_test.append(outputs.detach().cpu().numpy())
        
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net.classify(images)
            all_labels_train.append(labels.detach().cpu().numpy())
            all_outputs_train.append(outputs.detach().cpu().numpy())

    all_labels_test = reduce(lambda x,y: np.concatenate((x,y)), all_labels_test)
    all_outputs_test = reduce(lambda x,y: np.concatenate((x,y)), all_outputs_test)
    all_labels_train = reduce(lambda x,y: np.concatenate((x,y)), all_labels_train)[:all_labels_test.shape[0]]
    all_outputs_train = reduce(lambda x,y: np.concatenate((x,y)), all_outputs_train)[:all_labels_test.shape[0]]
    attack_dataset_test = [(all_labels_test[i],all_outputs_test[i],0) for i in range(all_labels_test.shape[0])]
    attack_dataset_train = [(all_labels_train[i],all_outputs_train[i],1) for i in range(all_labels_train.shape[0])]
    attack_dataset = attack_dataset_test + attack_dataset_train
    
    random.shuffle(attack_dataset)
    print('Total pairs generated:', len(attack_dataset))
    
    with open(save_attack_data, 'wb') as f:
        pickle.dump(attack_dataset, f)

def gen_meminf_dir(target_dir):
    model_list = os.listdir(target_dir)
    for mod in tqdm(model_list):
        # print(os.path.join(target_dir, mod))
        load_from = os.path.join(target_dir, mod, 'best_valid_ckpt.pt')
        gen_meminf(load_from)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Generator for membership inference dataset")
    parser.add_argument("--load_from", type=str, default='models/cifar')
    parser.add_argument("--save_attack_data", type=str)
    parser.add_argument("--target_dir", type=str)
    args = parser.parse_args()

    if args.target_dir is None:
        gen_meminf(args.load_from, args.save_attack_data)
    else:
        gen_meminf_dir(args.target_dir)

    