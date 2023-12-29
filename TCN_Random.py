import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import random
from tqdm import tqdm
import mmap
import struct
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import FCN, OS_CNN, OS_CNN_Trans, CNN_Trans, TCN
torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 以类的方式定义参数，模型超参数和一些其他设置


class Args:
    def __init__(self) -> None:
        self.batch_size = 512
        self.lr = 0.001  # 0.001 更好
        self.epochs = 100
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.window = 100  # 数据集长度 即window个长度的序列去预测下一个
        self.slide = 1  # 生成数据集时的滑动步长
        self.last_model_name = r"model\TCN_lastModel.pth"  # 最后的模型保存路径
        self.best_model_name = r"model\TCN_bestModel.pth"  # 最好的模型保存路径
        self.file_name = r"dataset\PRNG_2_23_10M.bin"  # 文件路径

        self.early_stop_epochs = 50  # 验证五次正确率无提升即停止训练
        # 随机选择0-300之间的数 按照print_idx==idx打印一下中间结果 这里300表示训练集或验证集最大的批次数
        self.print_idx = np.random.randint(0, 300, 1).item()
        self.prior = 0.0039
        ########################################################################################
        self.figPlot_path = r"log\TCN.svg"  # 修改模型时 一定要修改plot！！！！！！！！#
        ########################################################################################

# seed setting


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("cuda可用, 并设置相应的随机种子")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    torch.backends.cudnn.benchmark = False
    # if True, causes cuDNN to only use deterministic convolution algorithms.
    torch.backends.cudnn.deterministic = True

# 成批读取数据 GPU，CPU占用率都较高


class My_Dataset(Dataset):
    def __init__(self, data_path, batch_size, args):
        self.data_path = data_path
        self.batch_size = batch_size
        self.args = args
        with open(data_path, 'rb') as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.num_data = (len(self.mm) // 8 - args.window) // args.slide
        self.indices = list(range(self.num_data))  # 数据的索引列表

    def __getitem__(self, index):
        # 根据索引计算数据批次的起始和结束位置
        batch_indices = self.indices[index *
                                     self.batch_size: (index + 1) * self.batch_size]
        x_values = []
        y_values = []
        for idx in batch_indices:
            offset = idx * 8 * self.args.slide
            x_value = []
            for i in range(1, self.args.window + 1):
                x_data = self.mm[offset + (i - 1) * 8: offset + i * 8]
                x_value.append(struct.unpack('q', x_data)[0])  # 直接将数据转换为整数
            y_data = self.mm[offset + 8 *
                             self.args.window: offset + 8 * (self.args.window + 1)]
            y_value = struct.unpack('q', y_data)[0]  # 直接将数据转换为整数
            x_values.append(x_value)
            y_values.append(y_value)

        X = torch.from_numpy(
            np.array(x_values, dtype=np.int64))  # 从NumPy数组创建张量
        Y = torch.from_numpy(
            np.array(y_values, dtype=np.int64))  # 从NumPy数组创建张量
        return X, Y

    def __len__(self):
        return (self.num_data + self.batch_size - 1) // self.batch_size  # 计算数据集的长度

    def close(self):
        self.mm.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['mm']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        with open(self.data_path, 'rb') as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


class Trainer():
    def __init__(self, args, train_loader, val_loader, model):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.train_epochs_loss = []
        self.valid_epochs_loss = []
        self.train_acc = []
        self.val_acc = []

    def train(self):
        acc_max = self.args.prior
        stop_loop = False
        flag = 0
        same_seeds(2023)
        print(self.model)
        parameters = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        # 打印模型参数量
        print('Total parameters: {}'.format(parameters))
        # 可以添加label_smoothing参数 如0.1 是一种正则化的方法 防止过拟合 增强模型泛化能力
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)  # lr=0.01
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=0.00001, threshold=0.01,
                                      threshold_mode='abs', cooldown=5)

        for epoch in range(self.args.epochs):
            if stop_loop:  # 停止训练 start Ploting
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    "--------------------------------------------Training------------------------------------------------")
                progress_bar = tqdm(
                    self.train_loader, position=0, leave=True, total=len(self.train_loader))
                self.plot()

            self.model.train()
            train_epoch_loss = []
            acc = 0
            nums = 0
            # =========================train=======================

            for idx, (x, label) in enumerate(progress_bar):
                optimizer.zero_grad()
                #                 x = x.squeeze().to(self.args.device, non_blocking=True)
                #                 label = label.squeeze().to(self.args.device, non_blocking=True)
                # 这里的squeeze主要是去掉 设置的batch_size 1  1*512*100
                x = x.squeeze_().to(self.args.device, non_blocking=True)
                label = label.squeeze_().to(self.args.device, non_blocking=True)
                out = self.model(x)
                loss = criterion(out, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
                optimizer.step()

                train_epoch_loss.append(loss.detach().item())

                # acc的计算
                #                 out = F.softmax(out, dim=1)
                max_index = torch.argmax(out, dim=1)
                acc += sum(torch.eq(max_index, label)).cpu().item()
                # print(acc)
                nums += label.size()[0]
                train_accc = 100 * acc / nums
                train_loss = np.average(train_epoch_loss)
                self.train_acc.append(train_accc)

                progress_bar.set_description(
                    f"Training   Epoch [{epoch + 1}/{self.args.epochs}]")
                progress_bar.set_postfix(
                    {'loss': train_loss, 'acc': train_accc / 100})
                if (epoch + 1) % 10 == 0 and (idx + 1) == self.args.print_idx:
                    print("\n预测值: ", max_index[:32])
                    print("标签值: ", label[:32])

            self.train_epochs_loss.append(train_loss)
            # Update learning rate
            scheduler.step(train_loss)
            ###############################################
            print("train acc = {:.3f}%, loss = {}".format(
                train_accc, train_loss))

            # =========================val=========================
            if (epoch + 1) % 2 == 0:  # 每训练两轮验证一次
                val_epoch_loss = []
                acc, nums = 0, 0
                print(
                    "---------------------------------------------Valid-----------------------------------------------")
                val_progress_bar = tqdm(
                    self.val_loader, position=0, leave=True, total=len(self.val_loader))
                self.model.eval()
                with torch.no_grad():
                    for idx, (x, label) in enumerate(val_progress_bar):
                        #                         x = x.squeeze().to(self.args.device)  # .to(torch.float)
                        #                         label = label.squeeze().to(self.args.device)
                        x = x.squeeze_().to(self.args.device, non_blocking=True)
                        label = label.squeeze_().to(self.args.device, non_blocking=True)
                        out = self.model(x)
                        loss = criterion(out, label)
                        scheduler.step(loss)

                        val_epoch_loss.append(loss.detach().item())
                        max_index = torch.argmax(out, dim=1)
                        acc += sum(torch.eq(max_index, label)).cpu().item()
                        nums += label.size()[0]
                        val_accc = 100 * acc / nums
                        val_loss = np.average(val_epoch_loss)
                        self.val_acc.append(val_accc)

                        val_progress_bar.set_description(
                            f"Valid   Epoch [{epoch + 1}/{self.args.epochs}]")
                        val_progress_bar.set_postfix(
                            {'loss': val_loss, 'acc': val_accc / 100})
                        if (idx + 1) == self.args.print_idx:
                            print("\n预测值: ", max_index[:32])
                            print("标签值: ", label[:32])

                self.valid_epochs_loss.append(val_loss)
                # Update learning rate
                scheduler.step(val_loss)  #
                ###################################################
                print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(
                    epoch + 1, val_accc, val_loss))
                if val_accc / 100 > acc_max:
                    acc_max = val_accc / 100
                    # 保存验证集上acc最好的模型
                    torch.save(self.model.state_dict(),
                               self.args.best_model_name)
                    print("保存模型... 在acc={:.3f}%".format(val_accc))
                    self.plot()
                    if acc_max > 0.9:
                        print("验证集上正确率大于90%了 没必要再继续了")
                        stop_loop = True
                else:
                    flag += 1
                if flag >= self.args.early_stop_epochs:
                    print("模型已经没有提升了 终止训练")
                    stop_loop = True
                print(
                    "==============================================End===========================================")
        # =========================save model=====================  训练结束后 保存最后的模型
        torch.save(self.model.state_dict(), self.args.last_model_name)
        self.plot()

    def plot(self):
        # =========================plot==========================
        print("\nPloting...")
        plt.figure(figsize=(14, 10))
        plt.subplot(221)
        plt.plot(self.train_epochs_loss[:])
        plt.title("train loss")
        plt.xlabel('epoch')

        plt.subplot(222)
        plt.plot(self.train_epochs_loss, '-o', label="train_loss")
        plt.plot(self.valid_epochs_loss, '-o', label="valid_loss")
        plt.title("epochs loss for train and valid")
        plt.xlabel('epoch')
        plt.legend()

        plt.subplot(223)
        plt.plot(self.train_acc[:])
        plt.xticks(rotation=45)
        plt.title("train acc")
        plt.ylabel(f'Probability(%)')
        plt.xlabel('iteration')

        plt.subplot(224)
        plt.plot(self.val_acc[:])
        plt.title("val acc")
        plt.ylabel(f'Probability(%)')
        plt.xlabel('iteration')

        #         plt.legend()
        plt.savefig(self.args.figPlot_path, format='svg', bbox_inches='tight')


#         plt.show()
def creat_loader(dataset):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, persistent_workers=True,
                              num_workers=2, pin_memory=True, prefetch_factor=4, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, persistent_workers=True,
                            num_workers=2, pin_memory=True, prefetch_factor=4, drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    args = Args()
    # num_workers=24 256>>3.12 val 14 利用率比较稳定 显存2G  512>>>  val  1024>>> 3.15 val 11 显存10G  2048>>>3.13   val 10  显存18.6 利用率会跳
    # num_workers=16 256>>> val 512>>>3.06 val 11 恒定 1024>>>3.05 val 11 显存10.5G GPU利用率基本恒定97+  2048>>> 3.05  val 11  显存18.9 GPU利用率会跳
    # num_workers=8 256>>>    val    512>>>2.56 val 11 1024>>> 2.56 val 11  不太稳定 2048>>>2.54 val 11   基本恒定 几次跳到85% 第二轮会跳9%
    # num_workers=4 256>>>    val    512>>> val  1024>>>2.50 val 11  2048>>>2.49   val 11
    # num_workers=2 256>>>    val    512>>> val  1024>>>2.48 val 10  2048>>>2.51   val 9
    # num_workers=1 256>>>    val    512>>> val  1024>>> val   2048>>>2.57   val 20

    # persistent_workers=True, prefetch_factor=4
    # num_workers=2 1024>>>2.52 val 14
    # num_workers=4 1024>>>2.51 val 14

    dataset = My_Dataset(args.file_name, 512, args)
    train_loader, val_loader = creat_loader(dataset)
    # model = CNN_Trans().cuda()
    # 2**32=4294967296 10M数据量
    model = TCN(input_size=32, output_size=256, num_channels=[
                128, 128, 128, 128], kernel_size=3, dropout=0.2).cuda()
    trainer = Trainer(args, train_loader, val_loader, model)
    trainer.train()
    trainer.plot()
    
    
    
