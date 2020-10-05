from utils.attack_model import MEMBERINF

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import numpy as np
import os
import sys

from train_attack_model import separate_dataset, get_batch_dataset

from tqdm import tqdm

def eval_dataset(net, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = torch.FloatTensor(data[0]).to(device), torch.LongTensor(data[1]).to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def eval_single_model(att_model_prefix, att_dataset_prefix=None, all_classes=True, target_class=0, eval_all=False):
    if att_dataset_prefix is None:
        if 'cifar10' in att_model_prefix:
            att_dataset_prefix = os.path.join('meminf_data', 'cifar10', att_model_prefix.split('/')[-1])
        else:
            att_dataset_prefix = os.path.join('meminf_data', 'svhn', att_model_prefix.split('/')[-1])

    else:
        att_dataset_prefix = att_dataset_prefix

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    _path = att_dataset_prefix.split('/')
    path = '/'.join(_path[:-1])
    prefix = _path[-1]
    train_path_list = map(lambda x: os.path.join(path,x), [i for i in os.listdir(path) if prefix in i])

    target_classes = [i for i in range(10)] if all_classes else [target_class]
    data_lists = separate_dataset(train_path_list)

    print('Attack model:', att_model_prefix)
    print('Evaluation data:', att_dataset_prefix)
    advantages = []
    accs = []
    latex = []
    for t_class in target_classes:
        net = MEMBERINF()
        net.load_state_dict(torch.load(att_model_prefix+str(t_class)+'_best'))
        net.to(device)

        train_data_list = data_lists[t_class]
        trainloader, testloader = get_batch_dataset(train_data_list, batch_size=batch_size)

        if eval_all:
            testloader = trainloader + testloader
        acc = eval_dataset(net, testloader)
        advantages.append(2 * acc - 100)
        accs.append(acc)
        print('Accuracy of class', t_class, ':', acc)
        latex.append(str(round(acc, 2)))
    print('Average inference accuracy:', np.mean(accs))
    latex.append(str(round(np.mean(accs), 2)))
    print('Average inference advantage:', np.mean(advantages))
    latex.append(str(round(np.mean(advantages), 2)))
    print(' & '.join(latex))


def eval_dir(att_model_dir):
    for model in tqdm(os.listdir(att_model_dir)):
        att_model_prefix = os.path.join(att_model_dir, model)
        if os.path.isdir(att_model_prefix):
            eval_single_model(att_model_prefix + '/' + model, att_dataset_prefix=None, all_classes=True, target_class=0, eval_all=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Evaluate attack models.")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--all_classes", action="store_true", help="eval all classes")
    parser.add_argument("--att_model_prefix", type=str, help='load attack models from')
    parser.add_argument("--att_dataset_prefix", type=str, help='attack model dataset')
    parser.add_argument("--eval", action="store_true", help="eval train+test")
    parser.add_argument("--att_model_dir", type=str, help='attack all models in this dir')

    parser.add_argument("--print_to_log", action="store_true", help="log file")

    args = parser.parse_args()

    if args.print_to_log:
        sys.stdout = open(f'{args.att_model_dir}/eval_log.txt', 'w')
    
    if args.att_model_dir is not None:
        eval_dir(args.att_model_dir)
    else:
        eval_single_model(args.att_model_prefix, att_dataset_prefix=args.att_dataset_prefix, all_classes=args.all_classes, target_class=args.target_class, eval_all=args.eval)

    