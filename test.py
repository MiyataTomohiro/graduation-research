import matplotlib.pyplot as plt
import numpy as np
import math
import os

from keras.layers import Dense, Flatten, Reshape
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image

img_rows = 128
img_cols = 128
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100


def build_generator(img_shape, z_dim):  # 生成器

    model = Sequential()

    # 全結合層
    model.add(Dense(128, input_dim=z_dim))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # tanh関数を使った出力層
    model.add(Dense(128 * 128 * 3, activation="tanh"))

    # 生成器の出力が画像サイズになるようにreshapeする
    model.add(Reshape(img_shape))

    return model


def build_discriminator(img_shape):  # 識別器
    model = Sequential()

    # 入力画像を一列に並べる
    model.add(Flatten(input_shape=img_shape))

    # 全結合層
    model.add(Dense(128))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # sigmoid関数を通して出力する
    model.add(Dense(1, activation="sigmoid"))

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    # 生成器と識別器の統合
    model.add(generator)
    model.add(discriminator)

    return model


# 識別器の構築とコンパイル
discriminator = build_discriminator(img_shape)
discriminator.compile(
    loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"]
)

# 生成器の構築
generator = build_generator(img_shape, z_dim)

# 生成器の構築中は識別器のパラメータを固定
discriminator.trainable = False

# 生成器の訓練のため、識別器は固定し、GANモデルの構築とコンパイルを行う
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    width, height = generated_images.shape[2:]
    combined_image = np.zeros(
        (height * rows, width * cols), dtype=generated_images.dtype
    )

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[
            width * i: width * (i + 1), height * j: height * (j + 1)
        ] = image[0, :, :]
    return combined_image


def train(iterations, batch_size, sample_interval):

    # データセットのロード
    X_train = []
    img_list = os.listdir(input_path)
    for img in img_list:
        image = img_to_array(
            load_img(os.path.join(input_path, img), target_size=img_shape)
        )
        # -1から1の範囲に正規化
        image = (image.astype(np.float32) - 127.5) / 127.5
        X_train.append(image)

    # 4Dテンソルに変換(データの個数, 128, 128, 3)
    X_train = np.array(X_train)

    num_batches = int(X_train.shape[0] / batch_size)

    for epoch in range(iterations):
        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(batch_size)])
            image_batch = X_train[index * batch_size: (index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose=0)

            # 生成画像を出力
            if index % sample_interval == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                Image.fromarray(image.astype(np.uint8)).save(
                    output_path + "%04d_%04d.png" % (epoch, index)
                )
            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(batch_size)])
            g_loss = gan.train_on_batch(noise, [1] * batch_size)
            print(
                "epoch: %d, batch: %d, g_loss: %f, d_loss: %f"
                % (epoch, index, g_loss, d_loss)
            )


# ハイパラメータの設定
iterations = 20000
batch_size = 128
sample_interval = 1000

# 入出力パスの設定
name = input("フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
output_path = os.path.join("/home/miyata/test/img/res", name)

# 設定した反復回数だけGANの訓練を行う
train(iterations, batch_size, sample_interval)

losses = np.array(losses)

# 生成器と識別器の学習損失をプロット
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

accuracies = np.array(accuracies)

# 識別器の精度をプロット
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
