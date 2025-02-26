# パッケージのimport
import os.path as osp
from PIL import Image
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from utils.data_augumentation2 import *
from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
from itertools import cycle
import numpy as np
import random
import re

def make_datapath_list(rootpath, phase):
    if phase == 'train':
        imgpath_template = osp.join(rootpath, 'train', '%s.tiff') #変更
        annopath_template = osp.join(rootpath, 'annotations/train_annotations', '%s.png') #変更

        # 訓練のファイルのID（ファイル名）を取得する
        train_id_names = osp.join(rootpath + 'segmentations/train.txt') #変更

        # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
        train_img_list = list()
        train_anno_list = list()
        train_id_list = list()
        train_time_list = list()

        for line in open(train_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            # ファイル名の下2桁を取得
            individual_id = file_id[-2:]
            match = re.search(r'T(\d+)', file_id)
            if match:
                t_value_train = match.group(1)  # "T"に続く数字部分を取得
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            train_img_list.append(img_path)
            train_anno_list.append(anno_path)
            train_id_list.append(individual_id)
            train_time_list.append(t_value_train)

        return train_img_list, train_anno_list, train_id_list, train_time_list
    
    elif phase == 'val':
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'val', '%s.tiff')
        annopath_template = osp.join(rootpath, 'annotations/val_annotations', '%s.png')

        # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
        val_id_names = osp.join(rootpath + 'segmentations/val.txt')

        # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
        val_img_list = list()
        val_anno_list = list()
        val_id_list = list()
        val_time_list = list()

        for line in open(val_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            individual_id = file_id[-2:]
            match = re.search(r'T(\d+)', file_id)
            if match:
                t_value_val = match.group(1)  # "T"に続く数字部分を取得
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)
            val_id_list.append(individual_id)
            val_time_list.append(t_value_val)

        return val_img_list, val_anno_list, val_id_list, val_time_list
    
    elif phase == 'test':
        # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
        imgpath_template = osp.join(rootpath, 'test', '%s.tiff')
        annopath_template = osp.join(rootpath, 'annotations/test_annotations', '%s.png')

        # 訓練のファイルのID（ファイル名）を取得する
        test_id_names = osp.join(rootpath + 'segmentations/test.txt')

        # テストデータの画像ファイルとアノテーションファイルへのパスリストを作成
        test_img_list = list()
        test_anno_list = list()
        test_id_list = list()
        test_time_list = list()


        for line in open(test_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            individual_id = file_id[-2:]
            match = re.search(r'T(\d+)', file_id)
            if match:
                t_value_test = match.group(1)  # "T"に続く数字部分を取得
            else:
                t_value_test = 1
            img_path = (imgpath_template % file_id)  # 画像のパス
            anno_path = (annopath_template % file_id)  # アノテーションのパス
            test_img_list.append(img_path)
            test_anno_list.append(anno_path)
            test_id_list.append(individual_id)
            test_time_list.append(t_value_test)


        return test_img_list, test_anno_list, test_id_list, test_time_list


def make_datapath_list_unlabel(rootpath, phase):
    if phase == 'unlabel':
        imgpath_template = osp.join(rootpath, 'unlabel', '%s.tiff') #変更

        # 訓練のファイルのID（ファイル名）を取得する
        unlabel_id_names = osp.join(rootpath + 'segmentations/unlabel.txt') #変更

        # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
        unlabel_img_list = list()
        unlabel_id_list = list()
        unlabel_time_list = list()

        for line in open(unlabel_id_names):
            file_id = line.strip()  # 空白スペースと改行を除去
            individual_id = file_id[-2:] # ファイル名の下2桁を取得
            match = re.search(r'T(\d+)', file_id)
            if match:
                t_value_unlabel = match.group(1)  # "T"に続く数字部分を取得
            img_path = (imgpath_template % file_id)  # 画像のパス
            unlabel_img_list.append(img_path)
            unlabel_id_list.append(individual_id)
            unlabel_time_list.append(t_value_unlabel)

        return unlabel_img_list, unlabel_id_list, unlabel_time_list


class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std): #スケーリングは削除する
        self.input_size = input_size
        self.color_mean = color_mean
        self.color_std = color_std
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.8, 1.3]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                Brightness(brightness_range=(0.8, 1.5)),
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'test': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class GroupedBatchSampler(Sampler):
    """
    同じラベルを持つデータを同じミニバッチにまとめるサンプラー

    Attributes
    ----------
    dataset : Dataset
        対象のデータセット
    batch_size : int
        ミニバッチのサイズ
    label_to_indices : dict
        各ラベルに対応するデータのインデックスリスト
    drop_last : bool
        Trueの場合、不完全なバッチを破棄する
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        """
        Parameters
        ----------
        dataset : Dataset
            対象のデータセット
        batch_size : int
            ミニバッチのサイズ
        drop_last : bool, optional (default=True)
            Trueの場合、バッチサイズに満たないバッチを破棄する
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # ラベルごとにデータをグループ化する
        self.label_to_indices = defaultdict(list)
        for idx, (_, _, label,_) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)


    def __iter__(self):
        """
        ラベルごとにミニバッチを返すイテレーター
        """
        batch_indices = []
        
        for label, indices in self.label_to_indices.items():
            # 指定されたバッチサイズに基づいてラベルごとに分割する
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                # drop_last=Trueの場合、バッチサイズに満たないものは破棄
                if len(batch) == self.batch_size or not self.drop_last:
                    batch_indices.append(batch)
        

        for batch in batch_indices:
            yield batch

    def __len__(self):
        """
        サンプル数を返す
        """
        if self.drop_last:
            # drop_last=Trueの場合は完全なバッチ数のみ
            return sum(len(indices) // self.batch_size for indices in self.label_to_indices.values())
        else:
            # drop_last=Falseの場合は不完全なバッチも含める
            return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.label_to_indices.values())


class Unlabel_GroupedBatchSampler(Sampler):
    """
    同じラベルを持つデータを同じミニバッチにまとめるサンプラー

    Attributes
    ----------
    dataset : Dataset
        対象のデータセット
    batch_size : int
        ミニバッチのサイズ
    label_to_indices : dict
        各ラベルに対応するデータのインデックスリスト
    drop_last : bool
        Trueの場合、不完全なバッチを破棄する
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        """
        Parameters
        ----------
        dataset : Dataset
            対象のデータセット
        batch_size : int
            ミニバッチのサイズ
        drop_last : bool, optional (default=True)
            Trueの場合、バッチサイズに満たないバッチを破棄する
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # ラベルごとにデータをグループ化する
        self.label_to_indices = defaultdict(list)
        for idx, (_, label,_) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)

        #print(self.label_to_indices)


    def __iter__(self):
        """
        ラベルごとにミニバッチを返すイテレーター
        """
        batch_indices = []
        max_batches = max(len(indices) for indices in self.label_to_indices.values())

        for i in range(0, max_batches, self.batch_size):
            for label, indices in self.label_to_indices.items():
                start = i
                end = i + self.batch_size
                batch = indices[start:end]
                if len(batch) == self.batch_size or not self.drop_last:
                    batch_indices.append(batch)
        
        for batch in batch_indices:
            yield batch

    def __len__(self):
        """
        サンプル数を返す
        """
        if self.drop_last:
            # drop_last=Trueの場合は完全なバッチ数のみ
            return sum(len(indices) // self.batch_size for indices in self.label_to_indices.values())
        else:
            # drop_last=Falseの場合は不完全なバッチも含める
            return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.label_to_indices.values())


class WDDD_WDDD2_Dataset(data.Dataset):
    """
    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, label_list, time_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.label_list = label_list
        self.time_list = time_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        label = self.label_list[index]
        time = self.time_list[index]
        return img, anno_class_img, label, time

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


class WDDD_WDDD2_Dataset_Unlabel(data.Dataset):
    
    def __init__(self, img_list, label_list, time_list, phase, transform):
        self.img_list = img_list
        self.label_list = label_list
        self.time_list = time_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img = self.pull_item(index)
        label = self.label_list[index]
        time = self.time_list[index]
        return img, label

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]


        # 3. 前処理を実施
        img = self.transform(self.phase, img)

        return img


class BatchDataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.input_size = input_size
        self.color_mean = color_mean
        self.color_std = color_std
        
    def __call__(self, phase, batch_imgs, batch_annos):
        if phase == 'train':
            # バッチ全体で共通のパラメータを生成
            #scale = np.random.uniform(0.8, 1.3)
            rotation_angle = np.random.uniform(-90, 90)
            #brightness_factor = np.random.uniform(0.8, 1.5)
            gamma_factor = np.random.uniform(0.0, 2.0)
            do_mirror = np.random.randint(2)
            #gaussian = np.random.uniform(0.01, 0.15)
            
            processed_imgs = []
            processed_annos = []
            
            for img, anno in zip(batch_imgs, batch_annos):
                # 各変換を順番に適用
                #img, anno = Scale.apply_scale(img, anno, scale)
                img, anno = RandomRotation.apply_rotation(img, anno, rotation_angle)
                #img, anno = Brightness.apply_brightness(img, anno, brightness_factor)
                img, anno = GammaCorrection.apply(img, anno, gamma_factor)
                #img, anno = GaussianNoise.apply_gaussian_noise(img, anno, mean=0, std=gaussian)

                
                if do_mirror:
                    img, anno = RandomMirror.apply_mirror(img, anno)
                
                img, anno = Resize.apply_resize(img, anno, self.input_size)
                img, anno = Normalize_Tensor.apply_normalize_tensor(img, anno, self.color_mean, self.color_std)
                
                processed_imgs.append(img)
                processed_annos.append(anno)
            
            return torch.stack(processed_imgs), torch.stack(processed_annos)
        
        else:  # val または test
            processed_imgs = []
            processed_annos = []
            
            for img, anno in zip(batch_imgs, batch_annos):
                img, anno = Resize.apply_resize(img, anno, self.input_size)
                img, anno = Normalize_Tensor.apply_normalize_tensor(img, anno, self.color_mean, self.color_std)
                
                processed_imgs.append(img)
                processed_annos.append(anno)
            
            return torch.stack(processed_imgs), torch.stack(processed_annos)


class Unlabel_BatchDataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.input_size = input_size
        self.color_mean = color_mean
        self.color_std = color_std
        
    def __call__(self, phase, batch_imgs):

        # バッチ全体で共通のパラメータを生成
        #scale = np.random.uniform(0.8, 1.3)
        rotation_angle = np.random.uniform(-90, 90)
        #brightness_factor = np.random.uniform(0.8, 1.5)
        gamma_factor = np.random.uniform(0.0, 2.0)
        do_mirror = np.random.randint(2) 
        #gaussian = np.random.uniform(0.01, 0.15)
        processed_imgs = []

        for img in batch_imgs:
            # 各変換を順番に適用
            
            #img = Unlabel_Scale.apply_scale(img, scale)
            img = Unlabel_RandomRotation.apply_rotation(img, rotation_angle)
            #img = Unlabel_Brightness.apply_brightness(img, brightness_factor)
            img = Unlabel_GammaCorrection.apply(img, gamma_factor)
            #img = Unlabel_GaussianNoise.apply_gaussian_noise(img, mean=0, std=gaussian)
            
            if do_mirror:
                img = Unlabel_RandomMirror.apply_mirror(img)

            img = Unlabel_Resize.apply_resize(img, self.input_size)
            img = Unlabel_Normalize_Tensor.apply_normalize_tensor(img, self.color_mean, self.color_std)
            
            processed_imgs.append(img)

        return torch.stack(processed_imgs)


class WDDD_WDDD2_Dataset2(data.Dataset):
    """
    バッチ処理に対応したデータセットクラス
    """
    def __init__(self, img_list, anno_list, label_list, time_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.label_list = label_list
        self.time_list = time_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # 画像とアノテーションを PIL Image として読み込む
        img = Image.open(self.img_list[index])
        anno = Image.open(self.anno_list[index])
        label = self.label_list[index]
        time = self.time_list[index]
        
        return img, anno, label, time


class Unlabel_WDDD_WDDD2_Dataset2(data.Dataset):
    """
    バッチ処理に対応したデータセットクラス
    """
    def __init__(self, img_list, label_list, time_list, phase, transform):
        self.img_list = img_list
        self.label_list = label_list
        self.time_list = time_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # 画像とアノテーションを PIL Image として読み込む
        img = Image.open(self.img_list[index])
        label = self.label_list[index]
        time = self.time_list[index]
        
        return img, label, time


class BatchCollator:
    """
    DataLoaderで使用するカスタムcollate_fn
    バッチ単位でデータ拡張を適用する
    """
    def __init__(self, transform, phase):
        self.transform = transform
        self.phase = phase
        
    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        annos = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        times = [item[3] for item in batch]
        
        # バッチ全体に対して変換を適用
        imgs_tensor, annos_tensor = self.transform(self.phase, imgs, annos)
        labels_tensor = torch.tensor([int(label) for label in labels])
        times_tensor = torch.tensor([int(time) for time in times])
        
        return imgs_tensor, annos_tensor, labels_tensor, times_tensor


class Unlabel_BatchCollator:
    """
    DataLoaderで使用するカスタムcollate_fn
    バッチ単位でデータ拡張を適用する
    """
    def __init__(self, transform, phase):
        self.transform = transform
        self.phase = phase
        
    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        times = [item[2] for item in batch]
        
        # バッチ全体に対して変換を適用
        imgs_tensor = self.transform(self.phase, imgs)
        labels_tensor = torch.tensor([int(label) for label in labels])
        times_tensor = torch.tensor([int(time) for time in times])

        return imgs_tensor, labels_tensor, times_tensor


def create_dataloader(dataset, batch_size, phase):
    """
    バッチ単位のデータ拡張を適用するDataLoaderを作成
    """
    sampler = GroupedBatchSampler(dataset, batch_size)
    collator = BatchCollator(dataset.transform, phase)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4
    )
    
    return loader


def create_unlabel_dataloader(dataset, batch_size, phase):
    """
    バッチ単位のデータ拡張を適用するDataLoaderを作成
    """
    sampler = Unlabel_GroupedBatchSampler(dataset, batch_size)
    collator = Unlabel_BatchCollator(dataset.transform, phase)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4
    )
    
    return loader

