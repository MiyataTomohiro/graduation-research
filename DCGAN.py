import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Reshape,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img

img_rows = 128
img_cols = 128
channels = 3

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100


def build_generator(z_dim):

    model = Sequential()

    # Reshape input into 32x32x256 tensor via a fully connected layer
    model.add(Dense(256 * 32 * 32, input_dim=z_dim))
    model.add(Reshape((32, 32, 256)))

    # Transposed convolution layer, from 32x32x256 into 64x64x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 64x64x128 to 64x64x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 64x64x64 to 128x128x3 tensor
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding="same"))

    # Output layer with tanh activation
    model.add(Activation("tanh"))

    return model


def build_discriminator(img_shape):

    model = Sequential()

    # Convolutional layer, from 128x128x3 into 64x64x32 tensor
    model.add(
        Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 64x64x32 into 32x32x64 tensor
    model.add(
        Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 32x32x64 tensor into 16x16x128 tensor
    model.add(
        Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same")
    )

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Flatten())
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
generator = build_generator(z_dim)

# 生成器の構築中は識別器のパラメータを固定
discriminator.trainable = False

# 生成器の訓練のため、識別器は固定し、GANモデルの構築とコンパイルを行う
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []


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

    # 本物の画像のラベルは全て1とする
    real = np.ones((batch_size, 1))

    # 偽の画像のラベルは全て0とする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  識別器の訓練
        # -------------------------

        # 本物の画像をランダムに取り出したバッチを作る
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # 偽の画像のバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 識別器の訓練
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  生成器の訓練
        # ---------------------

        # 偽の画像のバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 生成器の訓練
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # 訓練終了後に図示するために、損失と精度をセーブしておく
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 訓練の進捗を出力する
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (iteration + 1, d_loss, 100.0 * accuracy, g_loss)
            )

            # 生成された画像のサンプルを出力する
            sample_images(generator, iteration)


def sample_images(generator, epoch, image_grid_rows=10, image_grid_columns=10):

    # ランダムノイズのサンプリング
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ランダムノイズを使って画像を生成する
    gen_imgs = generator.predict(z)

    # 画像の画素値を[0, 1]の範囲にスケールする
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 画像をグリッドに並べる
    fig, axs = plt.subplots(
        image_grid_rows,
        image_grid_columns,
        figsize=(128, 128),
        sharey=True,
        sharex=True,
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を入力する
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis("off")
            cnt += 1

    fig.savefig(os.path.join(output_path, f"{epoch}.png"))
    plt.close()


# ハイパラメータの設定
iterations = 20000
batch_size = 128
sample_interval = 1000

# 入出力パスの設定
name = input("フォルダー名を入力>>")
input_path = os.path.join("/home/miyata/test/img", name)
output_path = os.path.join("/home/miyata/test/img/res", name)
img_path = "/home/miyata/test/img/x_train"

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
plt.savefig(os.path.join(output_path, "Training Loss.png"))


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
plt.savefig(os.path.join(output_path, "Discriminator Accuracy.png"))
