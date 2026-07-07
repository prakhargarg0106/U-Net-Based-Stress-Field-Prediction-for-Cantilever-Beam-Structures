#importing libraries
import os
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam

#getting data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "prob1_data")
ZIP_PATH = os.path.join(BASE_DIR, "prob1_data(1).zip")

MAX_SAMPLES = None #can give a max for testing purposes, None is for all 5k

# training settings
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

#image settings
IMG_SIZE = 64

# output has 226 volume/thickness values
#therefore,we pad 226 to 16 x 16 = 256 and upsample to 64 x 64
VOL_H = 16
VOL_W = 16

#each stress file has 3304 stress values
#3304 = 56 x 59: thus rehsape stress to 56 x 59 then pad to 64 x 64.
STRESS_H = 56
STRESS_W = 59

RANDOM_SEED = 42 #for testing purposes
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#extracting data
def extractData():
    #checking if data folder exits, if not, then extract it
    if os.path.exists(DATA_FOLDER):
        print("Using existing data folder:", DATA_FOLDER)
        return

    if not os.path.exists(ZIP_PATH): #error catching
        raise FileNotFoundError(
            "Could not find prob1_data folder or prob1_data(1).zip"
            "Keep the data zip in the same folder as this script"
        )

    print("Extracting data zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(BASE_DIR)

    print("Extraction complete")

#loading the volume/stress data
def loadVolData():
    #loading the output file where each row was a vloume fraction and column a thickness value
    path = os.path.join(DATA_FOLDER, "output.xlsx")
    df = pd.read_excel(path, header=None)

    volume = df.values.astype(np.float32)

    #remove the first row as it is serial no.s from 0-225
    volume = volume[1:]

    if MAX_SAMPLES is not None: #testing purposes
        volume = volume[:MAX_SAMPLES]

    print("Volume data shape:", volume.shape)
    return volume

#loading stress data
def fileNum(filename): #keeping files in the correct numerical order
    match = re.search(r"stress_(\d+)\.txt", filename)
    if match:
        return int(match.group(1))     
    else:
        return -1


def loadStressData():
    #getting Von Mises stresses, that is, column 1 of each stress file
    folder = os.path.join(DATA_FOLDER, "stress")

    files = [
        f for f in os.listdir(folder)
        if f.startswith("stress_") and f.endswith(".txt")
    ]
    files = sorted(files, key=fileNum)

    if MAX_SAMPLES is not None: #for testing purposes
        files = files[:MAX_SAMPLES]

    stress_list = []
    for i, filename in enumerate(files):
        path = os.path.join(folder, filename)
        data = np.loadtxt(path)
        stress = data[:, 0]
        stress_list.append(stress)

        if (i + 1) % 500 == 0:
            print("Loaded stress files:", i + 1)

    stress_array = np.array(stress_list, dtype=np.float32)
    print("Stress data shape:", stress_array.shape)
    return stress_array

#converitng the vector to images
def volumeVectorToImg(vector):
    #converting the 226 values to a 64x64 image
    padded = np.zeros(VOL_H * VOL_W, dtype=np.float32)
    padded[:len(vector)] = vector

    small_image = padded.reshape(VOL_H, VOL_W)

    scale = IMG_SIZE // VOL_H
    image = np.repeat(np.repeat(small_image, scale, axis=0), scale, axis=1)

    return image[..., np.newaxis]

#converting stress vector to images
def stressVectorToImg(vector):
    #converting the 3304 values to a 64x64 image
    stress_image = vector.reshape(STRESS_H, STRESS_W)

    image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    image[:STRESS_H, :STRESS_W] = stress_image

    return image[..., np.newaxis]

#min max normalization
def normalize(data):
    data_min = data.min()
    data_max = data.max()

    normalized = (data - data_min) / (data_max - data_min + 1e-8)

    return normalized, data_min, data_max

#building U Net
def convBlock(x, filters):
    #Convolution Block: Conv2D to COnv2D
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)

    return x


def buildUNet(input_shape=(64, 64, 1), filters=[16, 32, 64], learning_rate=0.001):
    inputs = Input(shape=input_shape)
    #encoder
    c1 = convBlock(inputs, filters[0])
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = convBlock(p1, filters[1])
    p2 = MaxPooling2D((2, 2))(c2)
    #Bottleneck
    c3 = convBlock(p2, filters[2])
    #Decoder
    u2 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding="same")(c3)
    u2 = concatenate([u2, c2])
    c4 = convBlock(u2, filters[1])
    u1 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding="same")(c4)
    u1 = concatenate([u1, c1])
    c5 = convBlock(u1, filters[0])

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse",metrics=["mae"])
    return model


#plot functions
def plotLoss(history, title):
    #training v validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plotField(true_field, predicted_field, title):
    #true field, predicted field and absolute eror
    error = np.abs(true_field - predicted_field)

    plt.figure(figsize=(6, 5))
    plt.imshow(true_field, aspect="auto")
    plt.colorbar()
    plt.title(title + " - True")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(predicted_field, aspect="auto")
    plt.colorbar()
    plt.title(title + " - Predicted")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(error, aspect="auto")
    plt.colorbar()
    plt.title(title + " - Absolute Error")
    plt.xlabel("X grid")
    plt.ylabel("Y grid")
    plt.show()

#printing the results
def printMetrics(y_true, y_pred, valid_h, valid_w, name):
    #mse and mae
    true_valid = y_true[:, :valid_h, :valid_w, :]
    pred_valid = y_pred[:, :valid_h, :valid_w, :]
    mse = np.mean((true_valid - pred_valid) ** 2)
    mae = np.mean(np.abs(true_valid - pred_valid))

    print("\n" + name)
    print("MSE:", mse)
    print("MAE:", mae)

#MAIN
if __name__ == "__main__":

    #loading the data
    extractData()
    volumeVectors = loadVolData()
    stressVectors = loadStressData()
    if volumeVectors.shape[0] != stressVectors.shape[0]:
        raise ValueError("Number of volume samples and stress samples do not match.")

    #vectors to image like arrays
    volume_images = np.array(
        [volumeVectorToImg(v) for v in volumeVectors],
        dtype=np.float32
    )

    stress_images = np.array(
        [stressVectorToImg(s) for s in stressVectors],
        dtype=np.float32
    )
    print("Volume image shape:", volume_images.shape)
    print("Stress image shape:", stress_images.shape)

    #normalize input and output
    volume_norm, volume_min, volume_max = normalize(volume_images)

    #log training to reduce values
    stress_log = np.log1p(stress_images)
    stress_norm, stress_min, stress_max = normalize(stress_log)

    #forward
    # input = volume distribution
    # output = normalized Von Mises stress field
    X_train, X_temp, y_train, y_temp = train_test_split(
        volume_norm,
        stress_norm,
        test_size=0.30,
        random_state=RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    #inverse
    #input = normalized stress field
    # output = volume distribution
    X_inv_train = y_train
    y_inv_train = X_train

    X_inv_val = y_val
    y_inv_val = X_val

    X_inv_test = y_test
    y_inv_test = X_test

#foward unet
    print("\nTraining Forward U-Net: volume distribution -> normalized stress field")

    forward_unet = buildUNet(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        filters=[16, 32, 64],
        learning_rate=LEARNING_RATE
    )

    history_forward = forward_unet.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

#inverse u net
    print("\nTraining Inverse U-Net: normalized stress field -> volume distribution")

    inverse_unet = buildUNet(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        filters=[16, 32, 64],
        learning_rate=LEARNING_RATE
    )

    history_inverse = inverse_unet.fit(
        X_inv_train,
        y_inv_train,
        validation_data=(X_inv_val, y_inv_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    #predictions
    stress_pred = forward_unet.predict(X_test, verbose=0)
    volume_pred = inverse_unet.predict(X_inv_test, verbose=0)

    #evaluation on normalized fields
    printMetrics(
        y_test,
        stress_pred,
        STRESS_H,
        STRESS_W,
        "Forward U-Net normalized stress metrics"
    )

    printMetrics(
        y_inv_test,
        volume_pred,
        IMG_SIZE,
        IMG_SIZE,
        "Inverse U-Net normalized volume metrics"
    )

    #plots
    plotLoss(history_forward, "Forward U-Net Loss")
    plotLoss(history_inverse, "Inverse U-Net Loss")
    sample = 0

    plotField(
        y_test[sample, :STRESS_H, :STRESS_W, 0],
        stress_pred[sample, :STRESS_H, :STRESS_W, 0],
        "Forward U-Net Normalized Stress Field"
    )

    plotField(
        y_inv_test[sample, :, :, 0],
        volume_pred[sample, :, :, 0],
        "Inverse U-Net Volume Distribution"
    )
