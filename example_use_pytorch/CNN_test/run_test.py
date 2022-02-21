# -*- coding: utf-8 -*-

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import tools
from configs.config import Config
from datasets import DataGenerator, MapStyleDataset
from model_beifen_jiangwei1 import DifferentCNNModel
# from model_beifen import DifferentCNNModel
# from model import DifferentCNNModel
import matplotlib.pyplot as plt

k = 6

args = Config('./configs/config.yml')
preprocessing_args = args.preprocessing_config
training_args = args.training_config

current_dir = os.path.dirname(os.path.abspath(__file__))


# =============================== Fix Seed =================================================
def fix_random_seed(seed):
    # Fix random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_random_seed(training_args.seed)

# =============================== Device ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# model = DifferentCNNModel(dropout_rate=training_args.dropout_rate)
# if torch.cuda.device_count() > 1
#     model = nn.DataParallel(model)


# =============================== Data ======================================================
data_path = os.path.join(current_dir, preprocessing_args.preprocessed_file_path)


# ======================== Model Definition =================================================

def init_model_criterion_optimizer():
    # Model
    model = DifferentCNNModel(dropout_rate=training_args.dropout_rate).to(device)

    # Loss and optimizer
    criterion_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=training_args.learning_rate)

    return model, criterion_fn, optimizer


model, criterion_fn, optimizer = init_model_criterion_optimizer()
model.train()

# optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.75)

model_dir = os.path.join(current_dir, training_args.model_dir)
tools.check_dir_is_clean(model_dir)

print('=' * 25, 'DifferentCNNModel SUMMARY ON TRAIN MODE', '=' * 25, '\n')
print('seed: {}'.format(training_args.seed))
print('device: {}'.format(device))
print('data path: {}'.format(data_path))
print('lr: {}'.format(training_args.learning_rate))
print('model path: {}'.format(model_dir))
print('\nmodel parameters')
for name, para in model.named_parameters():
    # note:: para.shape 和 para.size() 返回一样
    print('{}: {}'.format(name, para.shape))
print()
print('\n', '=' * 25, 'END', '=' * 25)

# ============================  Training Summary Writer ==============================================
summary_dir = os.path.join(current_dir, training_args.summary_dir)
tools.check_dir_is_clean(summary_dir)

writer = SummaryWriter(summary_dir)

print('\033[5;41;255muse tensorboard to visualize loss:\ntensorboard --logdir= {} --port 8123\nopen localhost:8123 in Chrome\033[0m'.format(summary_dir))

# =============================== Training The Model =================================================
ten_fold_test_accuracies = list()
for test_fold_idx in range(k):
    print('Fold: {}'.format(test_fold_idx))

    # 轮换不同fold，产生不同的训练数据和测试数据
    dg = DataGenerator(dir_path=data_path, test_fold_idx=test_fold_idx)
    train_X, train_Y, test_X, test_Y = dg.load_data()

    train_data_loader = torch.utils.data.DataLoader(dataset=MapStyleDataset(train_X, train_Y), batch_size=training_args.batch_size, shuffle=True, num_workers = 0)
    test_data_loader = torch.utils.data.DataLoader(dataset=MapStyleDataset(test_X, test_Y), batch_size=training_args.batch_size)

    num_iter = len(train_data_loader) * training_args.num_epoch
    progress_bar = tools.ProcessBar(0, num_iter)

    loss_count = []
    for epoch_idx in range(training_args.num_epoch):
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            # images=torch.tensor(np.random.random(size=(32,1,252)),dtype=torch.float32)
            # print(images.size())
            # assert 1==2
            # first_difference_signals = images[0]
            _difference_signals = images[1]
            # print(type(first_difference_signals))

            # first_difference_signals = first_difference_signals.to(device)
            _difference_signals = _difference_signals.to(device)

            labels = labels.to(device)

            # Forward pass
            # outputs = model(first_difference_signals, second_difference_signals)
            # outputs = model(first_difference_signals)  # gai
            # print(len(outputs))
            outputs = model(_difference_signals)

            loss = criterion_fn(outputs, labels)

            # Backpropagation and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
            # loss_count.append(loss)
            if (batch_idx + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_idx + 1, training_args.num_epoch, batch_idx + 1, len(train_data_loader), loss.item()))
                # writer.add_scalar(tag='losses/loss', scalar_value=loss.item(), global_step=progress_bar.progress)
                # progress_bar.update(loss.item())
                loss_count.append(loss)

        model_path = os.path.join(model_dir, '{}.pth'.format(epoch_idx))
        torch.save(model.state_dict(), model_path)
        # loss_count.append(loss)
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count, label='Loss')
    plt.legend()
    plt.show()

    # =============================== Test ============================================
    # 初始化测试model
    model, _, _ = init_model_criterion_optimizer()
    max_accuracy = -1

    acc_count = []
    for epoch_idx in range(training_args.num_epoch):
        model_path = os.path.join(model_dir, '{}.pth'.format(epoch_idx))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        correct = torch.zeros(1).squeeze().cpu()
        total = torch.zeros(1).squeeze().cpu()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_data_loader):
                ####  yuan
                # print(type(images))
                # images = images.to(device)
                # labels = labels.to(device)
                #
                # # Forward pass
                # outputs = model(images)
                ######

                # first_difference_signals = images[0]
                # print(len(first_difference_signals))
                _difference_signals = images[1]
                # print(type(first_difference_signals))

                # first_difference_signals = first_difference_signals.to(device)
                _difference_signals = _difference_signals.to(device)

                labels = labels.to(device)

                # Forward pass
                outputs = model(_difference_signals)
                # outputs = model(first_difference_signals)   # gai

                prediction = torch.argmax(outputs, 1)
                correct += (prediction == labels).cpu().sum().float()
                total += len(labels)

        accuracy = (correct / total).cpu().detach().data.numpy()
        writer.add_scalar(tag='eval/accuracy', scalar_value=accuracy, global_step=epoch_idx)
        acc_count.append(accuracy)


        # max_accuracy = max_accuracy if max_accuracy > accuracy else accuracy
        print('test accuracy: {:.3f}\n'.format(accuracy))
    plt.figure('PyTorch_CNN_acc')
    plt.plot(acc_count, label='acc')
    plt.legend()
    # plt.ioff()
    plt.show()
    ten_fold_test_accuracies.append(max_accuracy)

    # =============================== Train ===========================================
    # 初始化训练model
    model, criterion_fn, optimizer = init_model_criterion_optimizer()
    model.train()
ten_fold_mean_test_accuracy = np.mean(np.array(ten_fold_test_accuracies))
print('{} folds mean test accuracy is {:.3f}'.format(k, ten_fold_mean_test_accuracy))
