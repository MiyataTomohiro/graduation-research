# 画像処理モジュール "Pillow" で画像をリサイズする。
import glob
import os

from PIL import Image


# jpg形式ファイルの画像サイズを変更する
def resizeImage(inputImage, outputImage, filename, num):
    # 元画像読み込み
    img = Image.open(inputImage)
    # リサイズ
    image = img.resize(size=(128, 128), resample=Image.LANCZOS)
    outputPath = os.path.join(outputImage, f"{name}_{num:0>3}.jpg")
    # 画像の保存
    image.save(outputPath, quality=100)


# 入出力パスの設定
name = input()
output_path = os.path.join("C:/Users/Miyata Tomohiro/Desktop/5th/卒業研究/img/data", name)
n = 0

img_path = glob.glob("C:/Users/Miyata Tomohiro/Desktop/5th/卒業研究/img/makuhari/*.jpg")

for img in img_path:
    n += 1
    resizeImage(img, output_path, name, n)
