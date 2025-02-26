import optuna
import os
import time
import pandas as pd
import numpy as np
import statistics
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as func
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utils.DataLoader import make_datapath_list, DataTransform, WDDD_WDDD2_Dataset
from utils.model import UNet
from utils.seed import torch_fix_seed
from utils.calc_IoU import calculate_iou

from torch.optim.lr_scheduler import ReduceLROnPlateau


LOG_SAVE_PATH = "./log/"
WEIGHT_SAVE_PATH = "./weight/"
WEIGHT_NAME = "U-Net.pth"
CSV_NAME = "log_U-Net.csv"
SEED = 42
NUM_CLASSES = 3
NUM_EPOCH = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_fix_seed(SEED)
rootpath = "/mnt/c/dataset/wddd2_dataset/"
p_palette = [0, 0, 0, 255, 255, 255, 243, 152, 0]
color_mean = (0)
color_std = (1)
#color_mean = (0.485, 0.456, 0.406)
#color_std = (0.229, 0.224, 0.225)
batch_size = 8

train_img_list, train_label_list = make_datapath_list(rootpath=rootpath, phase='train')
train_dataset = WDDD_WDDD2_Dataset(train_img_list, train_label_list, phase='train', 
                                   transform=DataTransform(
                                       input_size=256, color_mean=color_mean, color_std=color_std))

val_img_list, val_label_list = make_datapath_list(rootpath=rootpath, phase='val')
val_dataset = WDDD_WDDD2_Dataset(val_img_list, val_label_list, phase='val', 
                                 transform=DataTransform(
                                     input_size=256, color_mean=color_mean, color_std=color_std))

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

net = UNet(in_ch=1, out_ch=3)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)  # Adamの推奨ハイパーパラメータを使用
#optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

writer = SummaryWriter(log_dir=LOG_SAVE_PATH)
def writer_scaler(epoch, train_IoU, train_loss, val_IoU, val_loss):
    writer.add_scalar('train/loss', train_loss, epoch+1)
    writer.add_scalar('train/class1_IoU', train_IoU['1'], epoch+1)
    writer.add_scalar('train/class2_IoU', train_IoU['2'], epoch+1)
    writer.add_scalar('train/mIoU', train_IoU['mIoU'], epoch+1)
    writer.add_scalar('val/loss', val_loss, epoch+1)
    writer.add_scalar('val/class1_IoU', val_IoU['1'], epoch+1)
    writer.add_scalar('val/class2_IoU', val_IoU['2'], epoch+1)
    writer.add_scalar('val/mIoU', val_IoU['mIoU'], epoch+1)

def color_label(img):
    label_img = Image.fromarray(np.uint8(img), mode='P')
    label_img.putpalette(p_palette)
    label_img = label_img.convert('RGB')
    return np.array(label_img)

def writer_image(phase, epoch, image, label, pred):
    for i in range(label.shape[0]):
        if (i % 8 == 0):
            label_img = color_label(label[i])
            pred_img = color_label(pred[i])
            writer.add_image(phase+'_image/'+str(epoch+1)+'_'+str(i)+'_'+'image', image[i], epoch)
            writer.add_image(phase+'_image/'+str(epoch+1)+'_'+str(i)+'_'+'label', label_img, epoch, dataformats='HWC')
            writer.add_image(phase+'_image/'+str(epoch+1)+'_'+str(i)+'_'+'pred', pred_img, epoch, dataformats='HWC')



def train_model(net, dataloaders_dict, optimizer, num_epochs):
    IoU_MAX = 0.0
    best_val_loss = float('inf')  # best_val_lossを無限大で初期化
    logs = []
    t_start = time.time()
    print("使用するデバイス：", device)
    net.to(device)

    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_train_loss, epoch_val_loss = [], []
        t_1_IoU_list, t_2_IoU_list, t_mIoU_list, t_mIoU_backin_list, v_1_IoU_list, v_2_IoU_list, v_mIoU_list, v_mIoU_backin_list = [], [], [], [], [], [], [], []

        print('-----------------------------')
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-----------------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('(train)')
            else:
                net.eval()
                print('-----------------------------')
                print('(val)')

            count = 0

            for img, label in tqdm(dataloaders_dict[phase]):
                img = img.to(device)
                label = label.to(device, dtype=torch.long)

                with torch.set_grad_enabled(phase == 'train'):
                    output = net(img)
                    pred = torch.argmax(output, dim=1)

                    all_loss = loss(output, label)
                    IoUs = calculate_iou(pred, label, NUM_CLASSES)

                    if phase == 'train':
                        all_loss.backward()  # 勾配の計算
                        optimizer.step()
                        optimizer.zero_grad()
                        t_1_IoU_list.append(IoUs[1])
                        t_2_IoU_list.append(IoUs[2])
                        t_mIoU_list.append(IoUs[3])
                        t_mIoU_backin_list.append(IoUs[4])
                        epoch_train_loss.append(all_loss.item())
                    else:
                        v_1_IoU_list.append(IoUs[1])
                        v_2_IoU_list.append(IoUs[2])
                        v_mIoU_list.append(IoUs[3])
                        v_mIoU_backin_list.append(IoUs[4])
                        epoch_val_loss.append(all_loss.item())

                    count += 1

        t_1_IoU_mean = statistics.mean(t_1_IoU_list)
        t_2_IoU_mean = statistics.mean(t_2_IoU_list)
        t_mIoU_mean = statistics.mean(t_mIoU_list)
        t_mIoU_backin_mean = statistics.mean(t_mIoU_backin_list)
        t_IoU_dict = {'1': t_1_IoU_mean, '2': t_2_IoU_mean, 'mIoU': t_mIoU_mean, 'mIoU_back': t_mIoU_backin_mean}
        v_1_IoU_mean = statistics.mean(v_1_IoU_list)
        v_2_IoU_mean = statistics.mean(v_2_IoU_list)
        v_mIoU_mean = statistics.mean(v_mIoU_list)
        v_mIoU_backin_mean = statistics.mean(v_mIoU_backin_list)
        v_IoU_dict = {'1': v_1_IoU_mean, '2': v_2_IoU_mean, 'mIoU': v_mIoU_mean, 'mIoU_back': v_mIoU_backin_mean}
        train_loss_mean = sum(epoch_train_loss) / len(epoch_train_loss)
        val_loss_mean = sum(epoch_val_loss) / len(epoch_val_loss)

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean

        if v_mIoU_mean > IoU_MAX:
            IoU_MAX = v_mIoU_mean
            weight_path = os.path.join(WEIGHT_SAVE_PATH, WEIGHT_NAME)
            torch.save(net.state_dict(), weight_path)
            print('-----------------------------')
            print('weight更新')

        elif epoch % 100 == 0:  # 100エポックごと
            weight_path = os.path.join(WEIGHT_SAVE_PATH, f"U-Net_epoch{epoch}.pth")  # ファイル名にエポック番号を含める
            torch.save(net.state_dict(), weight_path)
            print('-----------------------------')
            print(f'weight_epoch{epoch}保存')

        writer_scaler(epoch, t_IoU_dict, train_loss_mean, v_IoU_dict, val_loss_mean)

        print('-----------------------------')
        print('epoch {} || train_Loss: {:.4f} || train_mIoU: {:.4f} || val_Loss: {:.4f} || val_mIoU: {:.4f}'.format(
            epoch + 1, train_loss_mean, t_IoU_dict['mIoU'], val_loss_mean, v_IoU_dict['mIoU']))
        print('-----------------------------')
        t_epoch_finish = time.time()
        print("1epoch_time: {:.4f}sec.".format(t_epoch_finish - t_epoch_start))

        log_epoch = {'epoch': epoch + 1, 'train_loss': train_loss_mean, 'train_nucleus_IoU': t_1_IoU_mean,
                     'train_embryo_IoU': t_2_IoU_mean, 'train_mIoU': t_mIoU_mean, 'train_mIoU_backin': t_mIoU_backin_mean,
                     'val_loss': val_loss_mean, 'val_nucleus_IoU': v_1_IoU_mean, 'val_embryo_IoU': v_2_IoU_mean,
                     'val_mIoU': v_mIoU_mean, 'val_mIoU_backin': v_mIoU_backin_mean}

        if epoch % 100 == 0:
            img = img.detach().to('cpu').numpy()
            anno_img = label.detach().to('cpu').numpy()
            pred = pred.detach().to('cpu').numpy()
            writer_image(phase, epoch, img, anno_img, pred)
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(CSV_NAME)
    t_finish = time.time()
    print('timer: {:.2f}sec.'.format(t_finish - t_start))
    writer.close()
    
    return best_val_loss, IoU_MAX


def objective(trial):
    # ハイパーパラメータのサンプリング
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 8)

    # データローダの再定義（バッチサイズをサンプリングした値に変更）
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # モデル、損失関数、オプティマイザの定義
    net = UNet(in_ch=1, out_ch=3)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # モデルの訓練
    best_val_loss, _ = train_model(net, dataloaders_dict, optimizer, 10)  # 10エポックだけ訓練して評価

    return best_val_loss  # 目的関数は最小化を目指すので、検証損失を返す


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 最良のハイパーパラメータを使ってモデルを再訓練
    best_params = trial.params
    net = UNet(in_ch=1, out_ch=3)
    optimizer = optim.Adam(net.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    batch_size = best_params["batch_size"]

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    train_model(net, dataloaders_dict, optimizer, NUM_EPOCH)
    print(best_params["lr"])
    print(weight_decay=best_params["weight_decay"])


