import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img

# 画像のパラメータ(サイズ、チャンネル数)
img_rows = 256
img_cols = 256
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)


# 画像を表示する関数
def show_imgs(imgs, row, col):
    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plt.imshow(img)
    plt.show()


def train(iterations, batch_size, sample_interval):

    # データセットのロード
    images = []
    img_list = os.listdir(input_path)
    for img in img_list:
        image = img_to_array(
            load_img(os.path.join(input_path, img), target_size=img_shape)
        )
        # -1から1の範囲に正規化
        image = (image.astype(np.float32) - 127.5) / 127.5
        images.append(image)

    # 画像の表示
    show_imgs(images, 5, 8)


# ハイパラメータの設定
iterations = 30000
batch_size = 40
sample_interval = 1000

# 入出力パスの設定
name = input("フォルダー名を入力>>")
input_path = os.path.join("C:/Users/Miyata Tomohiro/source/py/test/img", name)

# 決められた回数だけDCGANの訓練を反復する
train(iterations, batch_size, sample_interval)
