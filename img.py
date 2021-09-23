import glob
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 画像をロードする関数
def load_img(name):
    img = image.load_img(name, target_size=(128, 128))
    return image.img_to_array(img) / 255


# 画像を表示する関数
def show_imgs(imgs, row, col):
    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plt.imshow(img)
    plt.show()


"""ここからテキストの載っていたソースコード"""
# gen = ImageDataGenerator(
#     rescale=1 / 255.0,
#     rotation_range=90.0,
#     width_shift_range=1.0,
#     height_shift_range=0.5,
#     shear_range=0.8,
#     zoom_range=0.5,
#     horizontal_flip=True,
#     vertical_flip=True,
# )


# iters = gen.flow_from_directory(
#     "img", target_size=(128, 128), class_mode="binary", batch_size=5, shuffle=True
# )

# x_train_batch, y_train_batch = next(iters)

# print("shape of x_train_batch:", x_train_batch.shape)
# print("shape of y_train_batch:", y_train_batch.shape)
"""ここまでテキストの載っていたソースコード"""

"""ここから参考にしたサイトに載っていたソースコード"""
# datagen = ImageDataGenerator(
#     rotation_range=40.0, width_shift_range=0.3, height_shift_range=0.3, zoom_range=0.3
# )

# 全体を見る必要があるパラメータを使用する場合はfitが必要
# 今回は特に必要ないけど一応やっておく
# datagen.fit(imgs)
"""ここまで参考にしたサイトに載っていたソースコード"""


# 指定したバッチサイズの回数だけ画像を生成する関数
def create(batch_size):
    """ここから参考にしたサイトに載っていたソースコード"""
    # gen = datagen.flow(imgs, labels, batch_size=batch_size)
    for i in range(0, batch_size):
        # data = next(gen)
        # print("len:{}, labels:{}".format(len(data[0]), data[1]))
        # show_imgs(data[0], 1, len(data[0]))
        """ここまで参考にしたサイトに載っていたソースコード"""

        # 横浜と幕張の画像のpath
        yokohama_path = glob.glob("img/yokohama/*.jpg")
        makuhari_path = glob.glob("img/makuhari/*.jpg")

        # ディレクトリの中にある横浜と幕張の画像をランダムに抽出
        yokohama_random_image = random.choice(yokohama_path)
        makuhari_random_image = random.choice(makuhari_path)

        # 横浜と幕張の画像を1枚ずつロード
        imgs = np.array(
            [load_img(yokohama_random_image), load_img(makuhari_random_image)]
        )

        # 画像の表示
        show_imgs(imgs, 1, 2)


# 表示させる画像枚数を入力
create(int(input("表示する画像の枚数>>")))
