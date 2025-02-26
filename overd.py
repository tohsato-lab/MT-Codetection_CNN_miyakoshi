import os
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import binary_dilation

def extract_edges_with_transparency(img, edge_color=(255, 0, 0, 255), thickness=1):
    """
    画像からエッジを抽出し、透明な背景に指定した色で表現する関数。

    Parameters:
        img (Image.Image): 入力画像 (Pillow Imageオブジェクト)
        edge_color (tuple): エッジの色 (RGBA形式)
        thickness (int): 線の太さ

    Returns:
        Image.Image: 透明な背景を持つ色付きエッジ画像
    """
    # グレースケールに変換
    gray_img = img.convert("L")

    # エッジ検出
    edges = gray_img.filter(ImageFilter.FIND_EDGES)

    # エッジをNumPy配列に変換
    edge_array = np.array(edges, dtype=np.uint8)

    # エッジを2値化
    binary_edges = edge_array > 128  # しきい値を調整可能

    # 線の太さを調整
    if thickness > 1:
        for _ in range(thickness - 1):
            binary_edges = binary_dilation(binary_edges)

    # 透明な背景の作成
    color_array = np.zeros((binary_edges.shape[0], binary_edges.shape[1], 4), dtype=np.uint8)
    # エッジ部分のみに色を適用
    color_array[binary_edges] = edge_color

    return Image.fromarray(color_array, "RGBA")

def overlay_multiple_edges(base_img, *edge_images):
    
    # ベース画像をRGBAモードに変換
    result = base_img.convert("RGBA")

    # 各エッジ画像を順番に重ねる
    for edge_img in edge_images:
        edge_rgba = edge_img.convert("RGBA")
        # サイズが異なる場合はエッジ画像をリサイズ
        if result.size != edge_rgba.size:
            edge_rgba = edge_rgba.resize(result.size)
        # エッジを重ねる
        result = Image.alpha_composite(result, edge_rgba)

    return result

if __name__ == "__main__":
    # ディレクトリパス
    tiff_dir = "./data/tiff"  # TIFF画像のディレクトリ
    png_dir1 = "./data/anno"  # PNG画像のディレクトリ1
    png_dir2 = "./result"  # PNG画像のディレクトリ2
    save_dir = "./data/output"
    os.makedirs(save_dir, exist_ok=True)

    # TIFF画像のファイル名を取得 (拡張子なし)
    tiff_files = [os.path.splitext(f)[0] for f in os.listdir(tiff_dir) if f.endswith('.tiff')]

    # 画像処理ループ
    for tiff_file in tiff_files:
        tiff_path = os.path.join(tiff_dir, f"{tiff_file}.tiff")
        png_path1 = os.path.join(png_dir1, f"{tiff_file}.png")
        png_path2 = os.path.join(png_dir2, f"{tiff_file}.png")

        # 画像を読み込む
        png_image1 = Image.open(png_path1).convert("RGBA")
        png_image2 = Image.open(png_path2).convert("RGBA")
        base_image = Image.open(tiff_path).convert("RGBA")

        # ベース画像をPNG画像のサイズにリサイズ
        base_image_resized = base_image.resize(png_image1.size)

        # 1つ目のエッジを抽出（青色、線の太さ1）
        edge_image1 = extract_edges_with_transparency(
            png_image1,
            edge_color=(255, 255, 0, 255),  # 青色
            thickness=1
        )

        # 2つ目のエッジを抽出（赤色、線の太さ1）
        edge_image2 = extract_edges_with_transparency(
            png_image2,
            edge_color=(255, 0, 0, 255),  # 赤色
            thickness=1
        )

        # 複数のエッジをベース画像に重ねる
        result = overlay_multiple_edges(base_image_resized, edge_image1, edge_image2)

        # 結果を保存
        save_path = os.path.join(save_dir, f"{tiff_file}_overlay.png")  # 保存パスを生成
        result.save(save_path)