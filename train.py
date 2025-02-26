import os
import time
import pandas as pd
import numpy as np
import statistics
from PIL import Image
from tqdm import tqdm
from ramps import exp_rampup
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utils.DataLoader import *
from utils.model import CoDetectionCNN
from utils.seed import torch_fix_seed
from utils.calc_IoU import calculate_iou


LOG_SAVE_PATH = "./log/"
WEIGHT_SAVE_PATH = "./weight/"
WEIGHT_NAME = "U-Net.pth"
CSV_NAME = "log_U-Net.csv"
SEED = 42
NUM_CLASSES = 3
NUM_EPOCH = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_fix_seed(SEED)
rootpath = "/mnt/c/dataset/wddd2_dataset_1cell/"
rootpath_emboss = "/mnt/c/dataset/wddd2_dataset_1cell_emboss/"
p_palette = [0, 0, 0, 255, 255, 255, 243, 152, 0]
color_mean = (0)
color_std = (1)
#color_mean = (0.485, 0.456, 0.406)
#color_std = (0.229, 0.224, 0.225)
batch_size = 4

train_img_list, train_label_list, train_id_list, train_time_list = make_datapath_list(rootpath=rootpath, phase='train')
train_th_img, train_th_label, train_th_id, train_th_time = make_datapath_list(rootpath=rootpath_emboss, phase='train')
train_dataset = WDDD_WDDD2_Dataset2(train_img_list, train_label_list, train_id_list, train_time_list, phase='train', 
                                   transform=BatchDataTransform(
                                       input_size=256, color_mean=color_mean, color_std=color_std), 
                                       )

train_th_dataset = WDDD_WDDD2_Dataset2(train_th_img, train_th_label, train_th_id, train_th_time, phase='train', 
                                   transform=BatchDataTransform(
                                       input_size=256, color_mean=color_mean, color_std=color_std), 
                                       )

train_dataloader = create_dataloader(
    dataset=train_dataset,
    batch_size=batch_size + 1, 
    phase='train'
)

train_th_dataloader = create_dataloader(
    dataset=train_th_dataset,
    batch_size=batch_size + 1, 
    phase='train'
)

dataloader_iter = iter(train_th_dataloader)


val_img_list, val_label_list, val_id_list, val_time_list = make_datapath_list(rootpath=rootpath, phase='val')
val_dataset = WDDD_WDDD2_Dataset2(val_img_list, val_label_list, val_id_list, val_time_list, phase='val', 
                                 transform=BatchDataTransform(
                                     input_size=256, color_mean=color_mean, color_std=color_std), 
                                       )

val_dataloader = create_dataloader(
    dataset=val_dataset,
    batch_size=batch_size + 1,  
    phase='val'
)

unlabel_img_list, unlabel_id_list, unlabel_time_list = make_datapath_list_unlabel(rootpath=rootpath, phase='unlabel')
unlabel_th_img, unlabel_th_id, unlabel_th_time = make_datapath_list_unlabel(rootpath=rootpath_emboss, phase='unlabel')
unlabel_dataset = Unlabel_WDDD_WDDD2_Dataset2(unlabel_img_list, unlabel_id_list, unlabel_time_list, phase='unlabel', 
                                   transform=Unlabel_BatchDataTransform(
                                       input_size=256, color_mean=color_mean, color_std=color_std), 
                                       )

unlabel_th_dataset = Unlabel_WDDD_WDDD2_Dataset2(unlabel_th_img, unlabel_th_id, unlabel_th_time, phase='unlabel', 
                                   transform=Unlabel_BatchDataTransform(
                                       input_size=256, color_mean=color_mean, color_std=color_std), 
                                       )

unlabel_dataloader = create_unlabel_dataloader(
    dataset=unlabel_dataset,
    batch_size=batch_size + 1, 
    phase='unlabel'
)

unlabel_th_dataloader = create_unlabel_dataloader(
    dataset=unlabel_th_dataset,
    batch_size=batch_size + 1, 
    phase='unlabel'
)

unlabel_dataloader_iter = iter(unlabel_th_dataloader)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
#ラベルなしデータをまとめる
unlabel_dataloaders_dict = {"unlabel": unlabel_dataloader}

student_model = CoDetectionCNN(n_channels=1, n_classes=NUM_CLASSES, up_mode='upconv')
teacher_model = CoDetectionCNN(n_channels=1, n_classes=NUM_CLASSES, up_mode='upconv')
usp_weight = 0.01
ema_decay  = 0.99
ce_loss    = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=5e-6, weight_decay=5e-5) 

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

def cons_loss(logit1, logit2):
    assert logit1.size() == logit2.size()
    return F.mse_loss(logit1, logit2)

def update_ema(model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

global_step = 0
def train_model(student_model, teacher_model, dataloaders_dict, unlabel_dataloaders_dict, optimizer, num_epochs):
    global global_step
    IoU_MAX = 0.0
    logs = []
    t_start = time.time()
    print("使用するデバイス：", device)
    student_model.to(device)
    teacher_model.to(device)

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_train_loss, epoch_val_loss = [], []
        t_1_IoU_list, t_2_IoU_list, t_mIoU_list, t_mIoU_backin_list, v_1_IoU_list, v_2_IoU_list, v_mIoU_list, v_mIoU_backin_list = [], [], [], [], [], [], [], []
        
        print('-----------------------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-----------------------------')
        
        for phase in ['train',  'unlabel', 'val']:
            if phase == 'train':
                student_model.train()
                teacher_model.train()
                print('(train)')
            elif phase == 'val':
                student_model.eval()
                teacher_model.eval()
                print('-----------------------------')
                print('(val)')
            else :
                student_model.train()
                teacher_model.train()
                print('-----------------------------')
                print('(unlabel)')
                
            dataloader_iter = iter(train_th_dataloader)  
            if phase == 'train' or phase == 'val':
                for img, label, _, t in tqdm(dataloaders_dict[phase]):
                    if phase == 'train':
                        th_img, _, _, _ = next(dataloader_iter)
                        th_img1, th_img2 = th_img[:batch_size].to(device), th_img[1:].to(device)
                    
                    img1, img2 = img[:batch_size].to(device), img[1:].to(device)
                    label1, label2 = label[:batch_size].to(device, dtype=torch.long), label[1:].to(device, dtype=torch.long)

                    # === forward ===
                    with torch.set_grad_enabled(phase == 'train'):
                        output1, output2 = student_model(img1, img2)
                        pred = torch.argmax(output1, dim=1)
                        loss1 = ce_loss(output1, label1)
                        loss2 = ce_loss(output2, label2)
                        loss = loss1 + loss2 
                        # IoUの計算
                        IoUs = calculate_iou(pred, label1, NUM_CLASSES)

                        # === semi-supervised ===
                        if phase == 'train':
                            update_ema(student_model, teacher_model, ema_decay, global_step)
                        with torch.no_grad():
                            if phase == 'train':
                                ema_outputs1, ema_outputs2 = teacher_model(th_img1, th_img2)
                            else:
                                ema_outputs1, ema_outputs2 = teacher_model(img1, img2)
                            ema_outputs1 = ema_outputs1.detach()
                            ema_outputs2 =ema_outputs2.detach()

                        consistency_loss1 = cons_loss(output1, ema_outputs1)
                        consistency_loss2 = cons_loss(output2, ema_outputs2)
                        consistency_loss  = consistency_loss1 + consistency_loss2 
                        consistency_loss *= exp_rampup(100)(epoch)*usp_weight
                        loss += consistency_loss

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()  # 勾配の計算
                            optimizer.step()
                            global_step += 1
                            t_1_IoU_list.append(IoUs[1])
                            t_2_IoU_list.append(IoUs[2])
                            t_mIoU_list.append(IoUs[3])
                            t_mIoU_backin_list.append(IoUs[4])
                            epoch_train_loss.append(loss.item())
                        else:
                            v_1_IoU_list.append(IoUs[1])
                            v_2_IoU_list.append(IoUs[2])
                            v_mIoU_list.append(IoUs[3])
                            v_mIoU_backin_list.append(IoUs[4])
                            epoch_val_loss.append(loss.item())
            
            else : #=== training without label ===
                unlabel_dataloader_iter = iter(unlabel_th_dataloader)
                for img, _, t in tqdm(unlabel_dataloaders_dict[phase]):
                    th_img, _, _  = next(unlabel_dataloader_iter)
                    th_img1, th_img2 = th_img[:batch_size].to(device), th_img[1:].to(device)
                    img1, img2 = img[:batch_size].to(device), img[1:].to(device)
                    
                    #forward
                    output1, output2 = student_model(img1, img2)
                    update_ema(student_model, teacher_model, ema_decay, global_step)           
                    with torch.no_grad():
                        ema_outputs1, ema_outputs2 = teacher_model(th_img1, th_img2)
                        ema_outputs1 = ema_outputs1.detach()
                        ema_outputs2 = ema_outputs2.detach()
                    # === consistency loss ===
                    consistency_loss1 = cons_loss(output1, ema_outputs1)
                    consistency_loss2 = cons_loss(output2, ema_outputs2)
                    consistency_loss  = consistency_loss1 + consistency_loss2 
                    consistency_loss *= exp_rampup(100)(epoch)*usp_weight
                    
                    # backward
                    optimizer.zero_grad()
                    consistency_loss.backward()
                    optimizer.step()
                    
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
        train_loss_mean = sum(epoch_train_loss)/len(epoch_train_loss)
        val_loss_mean = sum(epoch_val_loss)/len(epoch_val_loss)
        

        if v_mIoU_mean > IoU_MAX:
            IoU_MAX = v_mIoU_mean
            weight_path = os.path.join(WEIGHT_SAVE_PATH, WEIGHT_NAME)
            torch.save(teacher_model.state_dict(), weight_path)
            print('-----------------------------')
            print('weight更新')

        elif epoch % 100 == 0:  # 100エポックごと
            weight_path = os.path.join(WEIGHT_SAVE_PATH, f"U-Net_epoch{epoch}.pth")  # ファイル名にエポック番号を含める
            torch.save(teacher_model.state_dict(), weight_path)
            print('-----------------------------')
            print(f'weight_epoch{epoch}保存')
            
        writer_scaler(epoch, t_IoU_dict, train_loss_mean, v_IoU_dict, val_loss_mean)
        
        print('-----------------------------')
        print('epoch {} || train_Loss: {:.4f} || train_mIoU: {:.4f} || val_Loss: {:.4f} || val_mIoU: {:.4f}'.format(
            epoch+1, train_loss_mean, t_IoU_dict['mIoU'], val_loss_mean, v_IoU_dict['mIoU']))
        print('-----------------------------')
        t_epoch_finish = time.time()
        print("1epoch_time: {:.4f}sec.".format(t_epoch_finish-t_epoch_start))
            
        log_epoch = {'epoch': epoch+1, 'train_loss': train_loss_mean, 'train_nucleus_IoU': t_1_IoU_mean, 'train_embryo_IoU': t_2_IoU_mean, 'train_mIoU': t_mIoU_mean, 'train_mIoU_backin': t_mIoU_backin_mean,
                     'val_loss': val_loss_mean, 'val_nucleus_IoU': v_1_IoU_mean, 'val_embryo_IoU': v_2_IoU_mean, 'val_mIoU': v_mIoU_mean, 'val_mIoU_backin': v_mIoU_backin_mean}
        
        if epoch % 100 == 0:
            img = img.detach().to('cpu').numpy()
            anno_img = label.detach().to('cpu').numpy()
            pred = pred.detach().to('cpu').numpy()
            if phase == 'train':
                writer_image(phase, epoch, img, anno_img, pred)
            else:
                writer_image(phase, epoch, img, anno_img, pred)
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        
        df.to_csv(CSV_NAME)
    t_finish = time.time()
    print('timer: {:.2f}sec.'.format(t_finish - t_start))
    writer.close()

if __name__ == "__main__":
    train_model(student_model, teacher_model, dataloaders_dict, unlabel_dataloaders_dict, optimizer, NUM_EPOCH)