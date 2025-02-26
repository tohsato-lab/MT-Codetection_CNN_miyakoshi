import os
import os.path as osp
import statistics
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.functional as func

from utils.DataLoader import *
from utils.model import CoDetectionCNN
from utils.seed import torch_fix_seed
from utils.calc_IoU import calculate_iou

# パラメータ (学習時の設定に合わせて調整)
SEED = 42
NUM_CLASSES = 3
WEIGHT_PATH = './weight/U-Net.pth' 
CSV_NAME = "result.csv"
color_mean = (0)
color_std = (1)
input_size = 256
rootpath = "/mnt/c/dataset/wddd2_dataset_1cell/"  
result_save_path = "./result/"  # 結果を保存するディレクトリ
result_save_path2 = "./result2/"  # 結果を保存するディレクトリ
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p_palette = [0, 0, 0, 255, 255, 255, 243, 152, 0]
# 乱数シード固定
torch_fix_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# テストデータセットの読み込み
test_img_list, test_label_list, test_id_list, test_time_list = make_datapath_list(rootpath=rootpath, phase="test")
test_dataset = WDDD_WDDD2_Dataset(
    test_img_list,
    test_label_list,
    test_id_list,
    test_time_list,
    phase="test",
    transform=DataTransform(input_size=input_size, color_mean=color_mean, color_std=color_std)
)

test_sampler = GroupedBatchSampler(test_dataset, batch_size=1)
test_dataloader = data.DataLoader(test_dataset, batch_sampler=test_sampler)

net = CoDetectionCNN(n_channels=1, n_classes=NUM_CLASSES, up_mode='upconv').to(device)
net.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))  # 適切なデバイスにロード
net.eval()

# 結果保存ディレクトリが存在しない場合は作成
os.makedirs(result_save_path, exist_ok=True)
os.makedirs(result_save_path2, exist_ok=True)

t = 0
logs = []
test_1_IoU_list, test_2_IoU_list, test_mIoU_list = [], [], []
# テストループ
img2 = None
label2 = None
for img, label, _, _ in tqdm(test_dataloader):  # tqdmで進捗表示
    if img2 == None:
        img2 = img
        label2 = label
    img = img.to(device)
    img2 = img2.to(device)

    label = label.to(device, dtype=torch.long)
    label2 = label2.to(device, dtype=torch.long)
    """
    temp = img
    img = img[:1, :, :, :]
    img2 = temp[1:, :, :, :]
    img = img.to(device)
    img2 = img2.to(device)
    temp = label
    label = label[:1, :, :]
    label2 = temp[1:, :, :]
    label = label.to(device, dtype=torch.long)
    label2 = label2.to(device, dtype=torch.long) """
    # 推論
    with torch.no_grad():  # テスト時は勾配計算不要
        output, output2 = net(img, img2)
        pred = torch.argmax(output, dim=1).squeeze(0)  # クラス予測を取得

    # IoU計算
    IoU = calculate_iou(pred, label, num_classes=3)
    test_1_IoU_list.append(IoU[1])
    test_2_IoU_list.append(IoU[2])
    test_mIoU_list.append(IoU[3])

    # 結果を辞書に追加
    log_IoU = {
        'class1_IoU': IoU[1],
        'class2_IoU': IoU[2],
        'mIoU': IoU[3],
    }
    logs.append(log_IoU)

    #バッチ内の画像処理
    pred = pred.detach().to('cpu').numpy()
    pred_img = Image.fromarray(np.uint8(pred), mode='P')
    pred_img.putpalette(p_palette)

    filename = f"pred_{t:03d}.png"
    pred_img.save(osp.join(result_save_path2, filename))

    filename = os.path.basename(test_img_list[t])  # t番目の画像のファイル名を取得
    filename = filename.split('.')[0] + '.png'  # 拡張子を.pngに変更
    save_path = osp.join(result_save_path, filename)  # 保存パスを生成
    pred_img.save(save_path)  # 画像を保存
    
    img2 = img
    label2 = label

    t += 1


df = pd.DataFrame(logs)
df.to_csv(CSV_NAME)

# 平均と標準偏差の出力
mIoU_mean = statistics.mean(test_mIoU_list)
mIoU_pst = statistics.pstdev(test_mIoU_list)
nucleus_mean = statistics.mean(test_1_IoU_list)
nucleus_pst = statistics.pstdev(test_1_IoU_list)
embryo_mean = statistics.mean(test_2_IoU_list)
embryo_pst = statistics.pstdev(test_2_IoU_list)

print(
    "mean_IoU: {:.4f} ± {:.4f}, nucleus_IoU: {:.4f} ± {:.4f}, embryo_IoU: {:.4f} ± {:.4f}".format(
        mIoU_mean, mIoU_pst, nucleus_mean, nucleus_pst, embryo_mean, embryo_pst
    )
)
