import os
import time
import pickle
import numpy as np
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils.attack_model import MEMBERINF

def separate_dataset(path_list):
    """
        Read attack data from a list of paths
    """
    cls_dict = {i:[] for i in range(10)}
    for path in path_list:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for p in data:
            label, logits, in_out = p
            cls_dict[label].append((logits, in_out))
    return cls_dict

def get_batch_dataset(data_list, batch_size=64):
    '''
        data_list: [(logits, label)]
    '''
    logits = [i[0] for i in data_list]
    labels = [i[1] for i in data_list]
    logits_batch = [np.stack(logits[i*batch_size:(i+1)*batch_size]) \
                    for i in range(int(len(data_list)/batch_size))]
    labels_batch = [np.stack(labels[i*batch_size:(i+1)*batch_size]) \
                    for i in range(int(len(data_list)/batch_size))]
    dataset = [(logits_batch[i], labels_batch[i]) for i in range(len(logits_batch))]
    train_set = dataset[:int(0.8*len(dataset))]
    test_set = dataset[int(0.8*len(dataset)):]
    return train_set, test_set

def train_single_attack_model(path_prefix, target_class, save_to, overwrite=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    _path = path_prefix.split('/')
    path = '/'.join(_path[:-1])
    prefix = _path[-1]
    path_list = map(lambda x: os.path.join(path,x), [i for i in os.listdir(path) if prefix in i])
    data_list = separate_dataset(path_list)[target_class]
    print('load from:')
    [print(i) for i in path_list]
    print('Total',len(data_list),'data points')
    trainloader, testloader = get_batch_dataset(data_list, batch_size=batch_size)

    net = MEMBERINF()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    if save_to is None:
        dataset_str = 'cifar10' if 'cifar10' in path_prefix else 'svhn'
        save_to_dir = os.path.join('attack_models', dataset_str, prefix)
        if not os.path.exists(save_to_dir):
            os.mkdir(save_to_dir)
        save_to = os.path.join(save_to_dir, prefix + str(target_class))
    else:
        save_to = save_to
    
    print('Target model:', path_prefix, target_class)
    if os.path.exists(save_to+'_final') and not overwrite:
        print('Exist:', save_to)
        return
    print('Save to:', save_to)
    
    best_acc = 0
    for epoch in range(200):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = torch.FloatTensor(data[0]).to(device), torch.LongTensor(data[1]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 50-1:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader[:10000]:
                images, labels = torch.FloatTensor(data[0]).to(device), torch.LongTensor(data[1]).to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Epoch:',epoch)
        print('Accuracy of the network on the train logits: %d %%' % (
            100 * correct / total))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = torch.FloatTensor(data[0]).to(device), torch.LongTensor(data[1]).to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Epoch:',epoch)
        print('Accuracy of the network on the test logits: %d %%' % (
            100 * correct / total))
        
        if best_acc < 100 * correct / total:
            best_acc = 100 * correct / total
            torch.save(net.state_dict(), save_to+'_best')
        


    print('Finished Training')
    torch.save(net.state_dict(), save_to+'_final')
    print('Best acc:', best_acc)

def gen_cmd(path_prefix, target_class):
    c = str(target_class)
    cmd = "Your shell command for training an attack model of label c. (e.g., \"python train_attack_model.py --target_class=\"+ c +\" --path_prefix=\" + path_prefix)"
    return cmd

def train_all_class(path_prefix):
    temp_script_path = os.path.join('temp_scripts', path_prefix.split('/')[-1] + '.sh')
    with open(temp_script_path, 'w') as fp:
        for i in range(10):
            cmd = gen_cmd(path_prefix, i)
            fp.write(cmd)
    
    return subprocess.call(["bash", temp_script_path])

def train_dir(m_dir):
    """
        e.g., dir='meminf_data/cifar10'
    """
    meminf_data_list = os.listdir(m_dir)
    path_prefixes = set([os.path.join(m_dir, '_'.join(p.split('.')[0].split('_')[:-1])) for p in meminf_data_list])
    for path_prefix in path_prefixes:
        train_all_class(path_prefix)
        prefix = path_prefix.split('/')[-1]
        dataset_str = 'cifar10' if 'cifar10' in path_prefix else 'svhn'
        save_to_dir = os.path.join('attack_models', dataset_str, prefix)

        num = 0
        for target_class in range(10):
            save_to = os.path.join(save_to_dir, prefix + str(target_class))
            if os.path.exists(save_to + '_final'):
                num += 1
        # print(prefix, num)

        while num < 10:
            print('Waiting...')
            num = 0
            for target_class in range(10):
                save_to = os.path.join(save_to_dir, prefix + str(target_class))
                if os.path.exists(save_to + '_final'):
                    num += 1
            time.sleep(60)

        print('Done:', path_prefix)
        # subprocess.call(['Your commands for closing screens, if used multiple screens'])
            


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train a binary classifier for a target class.")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--path_prefix", type=str)
    parser.add_argument("--train_all_dir", type=str)
    args = parser.parse_args()

    if args.train_all_dir:
        train_dir(args.train_all_dir)
    else:
        train_single_attack_model(args.path_prefix, args.target_class, args.save_to)