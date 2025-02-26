# 既存のクラスに静的メソッドを追加
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img

class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # アノテーションのリサイズ
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # input_sizeよりも短い辺はpaddingする
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))


            img = Image.new(img.mode, (width, height), (0))
            img.paste(img_original, (pad_width_left, pad_height_top))
            
            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0,0,0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img

class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        return self.apply_scale(img, anno_class_img, scale)

    @staticmethod
    def apply_scale(img, anno_class_img, scale):
        width = img.size[0]
        height = img.size[1]

        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((scaled_w, scaled_h), Image.NEAREST)

        # 画像を元の大きさに
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop((left, top, left+width, top+height))

        else:
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0))
            img.paste(img_original, (pad_width_left, pad_height_top))
            
            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0,0,0))
            anno_class_img.paste(anno_class_img_original, (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img

class Unlabel_Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        return self.apply_scale(img, scale)

    @staticmethod
    def apply_scale(img, scale):
        width = img.size[0]
        height = img.size[1]

        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # 画像を元の大きさに
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
        else:
            img_original = img.copy()
            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0))
            img.paste(img_original, (pad_width_left, pad_height_top))
        
        return img

class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        angle = np.random.uniform(self.angle[0], self.angle[1])
        return self.apply_rotation(img, anno_class_img, angle)

    @staticmethod
    def apply_rotation(img, anno_class_img, angle):
        img = img.rotate(angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(angle, Image.NEAREST)
        return img, anno_class_img

class Unlabel_RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        angle = np.random.uniform(self.angle[0], self.angle[1])
        return self.apply_rotation(img, angle)

    @staticmethod
    def apply_rotation(img, angle):
        img = img.rotate(angle, Image.BILINEAR)
        return img
    
class RandomMirror(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            return self.apply_mirror(img, anno_class_img)
        return img, anno_class_img

    @staticmethod
    def apply_mirror(img, anno_class_img):
        img = ImageOps.mirror(img)
        anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img

class Unlabel_RandomMirror(object):
    def __call__(self, img):
        if np.random.randint(2):
            return self.apply_mirror(img)
        return img
    @staticmethod
    def apply_mirror(img):
        img = ImageOps.mirror(img)
        return img

class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        return self.apply_resize(img, anno_class_img, self.input_size)

    @staticmethod
    def apply_resize(img, anno_class_img, input_size):
        # 元画像のアスペクト比を維持
        w, h = img.size
        if w > h:
            new_h = input_size
            new_w = int(w * (input_size / h))
        else:
            new_w = input_size
            new_h = int(h * (input_size / w))

        # リサイズ
        img = img.resize((new_w, new_h), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((new_w, new_h), Image.NEAREST)

        return img, anno_class_img
    
class Unlabel_Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img):
        
        return self.apply_resize(img, self.input_size)

    @staticmethod
    def apply_resize(img, input_size):
        
        img = img.resize((input_size, input_size), 
                         Image.BICUBIC)
        
        return img

class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):
        return self.apply_normalize_tensor(img, anno_class_img, self.color_mean, self.color_std)

    @staticmethod
    def apply_normalize_tensor(img, anno_class_img, color_mean, color_std):
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, color_mean, color_std)

        anno_class_img = np.array(anno_class_img)
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img

class Unlabel_Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img):
        return self.apply_normalize_tensor(img, self.color_mean, self.color_std)

    @staticmethod
    def apply_normalize_tensor(img, color_mean, color_std):
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, color_mean, color_std)


        return img

class Brightness(object):
    def __init__(self, brightness_range=(0.8, 1.5)):
        self.brightness_range = brightness_range

    def __call__(self, img, anno_class_img):
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        return self.apply_brightness(img, anno_class_img, brightness_factor)

    @staticmethod
    def apply_brightness(img, anno_class_img, brightness_factor):
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        return img, anno_class_img
    
class Unlabel_Brightness(object):
    def __init__(self, brightness_range=(0.8, 1.5)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        return self.apply_brightness(img, brightness_factor)

    @staticmethod
    def apply_brightness(img, brightness_factor):
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        return img
    
class GammaCorrection(object):
    """ランダムにガンマ補正を行うクラス"""
    def __init__(self, gamma_factor):
        self.gamma_factor = gamma_factor

    def __call__(self, img, anno_class_img):
        gamma_factor = np.random.uniform(self.gamma_factor[0], self.gamma_factor[1])  # self.gamma_factor を使用する
        return self.apply(img, anno_class_img, gamma_factor)
    
    @staticmethod
    def apply(img, anno_class_img, gamma_factor):  # gamma_range を gamma_factor に変更
        """
        画像とアノテーション画像を受け取り、画像のガンマ補正をランダムに変更して返す

        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像
            gamma_factor (float): ガンマ補正の係数

        Returns:
            PIL.Image: ガンマ補正後の画像
            PIL.Image: 変更なしのアノテーション画像 (ガンマ補正は影響しない)
        """

        # 画像をNumPy配列に変換
        img_np = np.array(img)
        max_val = np.max(img_np)  # 画像の最大値を取得

        # 最大値で正規化してからガンマ補正を適用
        img_np = max_val * (img_np / max_val) ** (1/gamma_factor)  
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # [0, 255]の範囲に戻す

        # NumPy配列をPIL画像に変換
        img = Image.fromarray(img_np)

        return img, anno_class_img
    
class Unlabel_GammaCorrection(object):
    """ランダムにガンマ補正を行うクラス"""
    def __init__(self, gamma_factor):
        self.gamma_factor = gamma_factor

    def __call__(self, img):
        gamma_factor = np.random.uniform(self.gamma_factor[0], self.gamma_factor[1])  # self.gamma_factor を使用する
        return self.apply(img, gamma_factor)
    
    @staticmethod
    def apply(img, gamma_factor):  # gamma_range を gamma_factor に変更
        """
        画像とアノテーション画像を受け取り、画像のガンマ補正をランダムに変更して返す

        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像
            gamma_factor (float): ガンマ補正の係数

        Returns:
            PIL.Image: ガンマ補正後の画像
            PIL.Image: 変更なしのアノテーション画像 (ガンマ補正は影響しない)
        """

        # 画像をNumPy配列に変換
        img_np = np.array(img)
        max_val = np.max(img_np)  # 画像の最大値を取得

        # 最大値で正規化してからガンマ補正を適用
        img_np = max_val * (img_np / max_val) ** (1/gamma_factor)  
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # [0, 255]の範囲に戻す

        # NumPy配列をPIL画像に変換
        img = Image.fromarray(img_np)

        return img

class GaussianNoise(object):
    """画像にガウシアンノイズを加えるクラス"""
    def __init__(self, mean=0.0, std_range=(0.01, 0.15)):
        """
        Args:
            mean (float): ノイズの平均値
            std_range (tuple): 標準偏差の範囲 (最小値, 最大値)
        """
        self.mean = mean
        self.std_range = std_range

    def __call__(self, img, anno_class_img):
        """
        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像
        Returns:
            PIL.Image: ガウシアンノイズを追加した画像
            PIL.Image: 変更なしのアノテーション画像
        """
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        return self.apply_gaussian_noise(img, anno_class_img, self.mean, std)

    @staticmethod
    def apply_gaussian_noise(img, anno_class_img, mean, std):
        """
        ノイズを加える静的メソッド
        Args:
            img (PIL.Image): 入力画像
            anno_class_img (PIL.Image): アノテーション画像
            mean (float): ノイズの平均値
            std (float): ノイズの標準偏差
        Returns:
            PIL.Image: ガウシアンノイズを追加した画像
            PIL.Image: 変更なしのアノテーション画像
        """
        img_np = np.array(img).astype(np.float32)  # NumPy配列に変換
        noise = np.random.normal(mean, std * 255, img_np.shape)  # ガウシアンノイズを生成
        img_np = img_np + noise  # ノイズを追加
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # 範囲を[0, 255]にクリップして型を変更
        img = Image.fromarray(img_np)  # PIL画像に戻す
        return img, anno_class_img

class Unlabel_GaussianNoise(object):
    """ラベルなしデータにガウシアンノイズを加えるクラス"""
    def __init__(self, mean=0.0, std_range=(0.01, 0.15)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, img):
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        return self.apply_gaussian_noise(img, self.mean, std)

    @staticmethod
    def apply_gaussian_noise(img, mean, std):
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(mean, std * 255, img_np.shape)
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        return img

