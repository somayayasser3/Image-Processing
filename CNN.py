import cv2
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import random
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf


def ReadFiles():
    directory = glob.glob(
        'E:/Bioninformatics/year4,sem1/Machine Learning and Bioinformatics/Assignments/Assignment2/asl_alphabet_train/*')
    imageNamesTrain = []
    labelOutputTrain = []
    for folder in directory:
        for file in glob.glob(folder + '/*.jpg'):
            # print(file)
            imageNamesTrain.append(file)
    # img_DF=pd.DataFrame(imageNamesTrain)
    # train_head=imageNamesTrain.head()
    imageNamesT = random.sample(imageNamesTrain, len(imageNamesTrain))
    # print(len(imageNamesT))
    # imageNamesT=imageNamesTrain
    imageTrain = imageNamesT[
                 :1000]  # imageNamesT[:10] + imageNamesT[80140: 80150] + imageNamesT[40140:40150] + imageNamesT[90120:90150]
    for image in imageTrain:
        labels = image.split("/")
        name = labels[-1].split("\\")[1]
        labelOutputTrain.append(name)

    print(labelOutputTrain)

    labelOutputTest = []
    imageNamesTest = []
    for file in glob.glob(
            'E:/Bioninformatics/year4,sem1/Machine Learning and Bioinformatics/Assignments/Assignment2/asl_alphabet_test/*.jpg'):
        imageNamesTest.append(file)

    for image in imageNamesTest:
        labels = image.split("/")[-1]
        name = labels.split("\\")[1]
        finLabel = name.split("_")[0]
        labelOutputTest.append(finLabel)
    TrainData = imageTrain
    TestData = imageNamesTest
    return TrainData, TestData, labelOutputTest, labelOutputTrain


def ReadImages(imageNames, vOc):
    readImagesGray = []
    readImagesRGB = []

    if vOc == "CNN":
        for image in imageNames:
            imgRGB = cv2.imread(image)
            imgRGB = cv2.resize(imgRGB, (24, 24))
            # imgRGB = cv2.resize(imgRGB, (0, 0), fx=0.25, fy=0.25)
            imgRGB = (imgRGB - np.min(imgRGB)) / (np.max(imgRGB) - np.min(imgRGB))  # image normalization
            readImagesRGB.append(imgRGB)

            imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            imgG = cv2.resize(imgG, (24, 24))
            # imgG = cv2.resize(imgG, (0, 0), fx=0.25, fy=0.25)
            imgG = (imgG - np.min(imgG)) / (np.max(imgG) - np.min(imgG))

            readImagesGray.append(imgG)
        readImagesRGB = np.array(readImagesRGB)
        readImagesGray = np.array(readImagesGray)
        readImagesGray = np.expand_dims(readImagesGray, axis=3)
        print(readImagesGray.shape)
    elif vOc == "VGG" or vOc == "res":
        for image in imageNames:
            imgRGB = cv2.imread(image)
            imgRGB = cv2.resize(imgRGB, (224, 224))
            # imgRGB = cv2.resize(imgRGB, (0, 0), fx=0.25, fy=0.25)
            imgRGB = (imgRGB - np.min(imgRGB)) / (np.max(imgRGB) - np.min(imgRGB))  # image normalization
            readImagesRGB.append(imgRGB)

            imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            imgG = cv2.resize(imgG, (224, 224))
            # imgG = cv2.resize(imgG, (0, 0), fx=0.25, fy=0.25)
            imgG = (imgG - np.min(imgG)) / (np.max(imgG) - np.min(imgG))

            readImagesGray.append(imgG)
        readImagesRGB = np.array(readImagesRGB)
        readImagesGray = np.array(readImagesGray)
        readImagesGray = np.expand_dims(readImagesGray, axis=3)
        print(readImagesGray.shape)

    return readImagesRGB, readImagesGray


def random_flip(img):
    if np.random.rand() > 0.7:
        return cv2.flip(img, 0)
    return cv2.flip(img, 1)


def rotate(img):
    height, width, _ = img.shape
    M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
    M2[0, 2] = -M2[0, 1] * width / 2
    M2[1, 2] = -M2[1, 0] * width / 2
    rotated_img = cv2.warpAffine(img, M2, (width, height))
    return rotated_img


def Augmentation(images):
    oldImages = images
    newImages = []
    for img in oldImages:
        img = random_flip(img)
        img = rotate(img)
        newImages.append(img)
    return newImages


def CNNModel1(Train, labelTrain, Test, labelTest, imageType):
    # batch_size = 64
    epochs = 10
    num_classes = 29
    num_filters = 25
    filter_size = 2
    pool_size = 2
    imageType = imageType.upper()

    if imageType == "GRAY":
        model = Sequential([
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),

            Flatten(),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(400, activation="LeakyReLU"),
            Dropout(0.3, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])

        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),

        )

    elif imageType == "RGB":
        model = Sequential([
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(num_filters, filter_size, strides=(2, 2), input_shape=(24, 24, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),

            Flatten(),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(400, activation="LeakyReLU"),
            Dropout(0.3, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])

        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )


def CNNModel2(Train, labelTrain, Test, labelTest, imageType):
    batch_size = 64
    epochs = 10
    num_classes = 29
    num_filters = 35
    filter_size = 3
    pool_size = 3
    imageType = imageType.upper()

    if imageType == "GRAY":
        model = Sequential([
            Conv2D(num_filters, filter_size, input_shape=(100, 100, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(3, 3)),
            Conv2D(num_filters, filter_size, input_shape=(100, 100, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(3, 3)),

            Flatten(),
            Dense(550, activation="relu"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(400, activation="relu"),
            Dropout(0.3, seed=2022),
            Dense(550, activation="relu"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(200, activation="relu"),
            Dropout(0.2, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])

        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
            batch_size=batch_size,
        )

    elif imageType == "RGB":
        model = Sequential([
            Conv2D(num_filters, filter_size, input_shape=(100, 100, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(3, 3)),
            Conv2D(num_filters, filter_size, input_shape=(100, 100, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(3, 3)),

            Flatten(),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(400, activation="LeakyReLU"),
            Dropout(0.3, seed=2022),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(200, activation="LeakyReLU"),
            Dropout(0.2, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])

        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
            batch_size=batch_size,
        )


# def VVG16(Train, Test):
#     model = VGG16(weights='imagenet', include_top=False)
#     img_path = Train
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features = model.predict(x)


def vgg16(Train, labelTrain, Test, labelTest, imageType):
    # batch_size = 64
    epochs = 8
    num_classes = 29
    filter_size = 3
    pool_size = 2
    imageType = imageType.upper()

    if imageType == "GRAY":
        # model = Sequential([
        #     Conv2D(32, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #
        #     Flatten(),
        #     Dense(550, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(550, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(550, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(num_classes, activation='softmax'),  # ouput layer
        # ])
        model = Sequential([
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(128, filter_size, input_shape=(112, 112, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(112, 112, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(256, filter_size, input_shape=(56, 56, 1), padding="same"),
            Conv2D(256, filter_size, input_shape=(56, 56, 1), padding="same"),
            Conv2D(256, filter_size, input_shape=(56, 56, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(512, filter_size, input_shape=(28, 28, 1), padding="same"),
            Conv2D(512, filter_size, input_shape=(28, 28, 1), padding="same"),
            Conv2D(512, filter_size, input_shape=(28, 28, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(512, filter_size, input_shape=(14, 14, 1), padding="same"),
            Conv2D(512, filter_size, input_shape=(14, 14, 1), padding="same"),
            Conv2D(512, filter_size, input_shape=(14, 14, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),

            Flatten(),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )

    elif imageType == "RGB":
        model = Sequential([
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(128, filter_size, input_shape=(112, 112, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(112, 112, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(256, filter_size, input_shape=(56, 56, 3), padding="same"),
            Conv2D(256, filter_size, input_shape=(56, 56, 3), padding="same"),
            Conv2D(256, filter_size, input_shape=(56, 56, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(512, filter_size, input_shape=(28, 28, 3), padding="same"),
            Conv2D(512, filter_size, input_shape=(28, 28, 3), padding="same"),
            Conv2D(512, filter_size, input_shape=(28, 28, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(512, filter_size, input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, filter_size, input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, filter_size, input_shape=(14, 14, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),

            Flatten(),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(4096, activation="relu"),  # Adding the Hidden layer
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        # model = Sequential([
        #     Conv2D(32, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(32, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #
        #     Flatten(),
        #     Dense(550, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(400, activation="relu"),
        #     Dropout(0.3, seed=2022),
        #     Dense(550, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     # Dense(200, activation="relu"),
        #     # Dropout(0.2, seed=2022),
        #     Dense(num_classes, activation='softmax'),  # ouput layer
        # ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )


def vgg19(Train, labelTrain, Test, labelTest, imageType):
    # batch_size = 64
    epochs = 5
    num_classes = 29
    # num_filters = 13
    filter_size = 3
    pool_size = 2
    imageType = imageType.upper()

    if imageType == "GRAY":
        model = Sequential([
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(256, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size),

            Flatten(),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )

    elif imageType == "RGB":
        model = Sequential([
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(64, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(128, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(256, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 3), padding="same"),
            Conv2D(256, filter_size, input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size),

            Flatten(),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(400, activation="LeakyReLU"),
            Dropout(0.3, seed=2022),
            Dense(550, activation="LeakyReLU"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            # Dense(200, activation="LeakyReLU"),
            # Dropout(0.2, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )


def res50(Train, labelTrain, Test, labelTest, imageType):
    # batch_size = 64
    epochs = 5
    num_classes = 29
    # filter_size = 3
    pool_size = 2
    imageType = imageType.upper()

    if imageType == "GRAY":
        model = Sequential([
            Conv2D(64, kernel_size=(7, 7), strides=(2, 2), input_shape=(224, 224, 1), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),

            Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),

            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),

            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),

            AveragePooling2D(pool_size=pool_size),

            Flatten(),
            Dense(1000, activation="relu"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        # model = Sequential([
        #     Conv2D(64, kernel_size=(7, 7), strides=(2, 2), input_shape=(224, 224, 1), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(224, 224, 1), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(224, 224, 1), padding="same"),
        #
        #     AveragePooling2D(pool_size=pool_size),
        #
        #     Flatten(),
        #     Dense(1000, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(num_classes, activation='softmax'),  # ouput layer
        # ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )

    elif imageType == "RGB":
        # model = Sequential([
        #     Conv2D(64, kernel_size=(7, 7), strides=(2, 2), input_shape=(224, 224, 3), padding="same"),
        #     MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
        #
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
        #     Conv2D(64, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
        #
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
        #
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
        #     Conv2D(128, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
        #
        #     AveragePooling2D(pool_size=pool_size),
        #
        #     Flatten(),
        #     Dense(1000, activation="relu"),  # Adding the Hidden layer
        #     Dropout(0.1, seed=2022),
        #     Dense(num_classes, activation='softmax'),  # ouput layer
        # ])
        model = Sequential([
            Conv2D(64, kernel_size=(7, 7), strides=(2, 2), input_shape=(224, 224, 3), padding="same"),
            MaxPooling2D(pool_size=pool_size, strides=(2, 2)),
            Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
            Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
            Conv2D(64, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),
            Conv2D(64, kernel_size=(3, 3), input_shape=(112, 112, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(112, 112, 3), padding="same"),

            Conv2D(128, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),
            Conv2D(128, kernel_size=(3, 3), input_shape=(56, 56, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(56, 56, 3), padding="same"),

            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),
            Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 3), padding="same"),
            Conv2D(1024, kernel_size=(1, 1), input_shape=(28, 28, 3), padding="same"),

            Conv2D(512, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),
            Conv2D(512, kernel_size=(3, 3), input_shape=(14, 14, 3), padding="same"),
            Conv2D(2048, kernel_size=(1, 1), input_shape=(14, 14, 3), padding="same"),

            AveragePooling2D(pool_size=pool_size),

            Flatten(),
            Dense(1000, activation="relu"),  # Adding the Hidden layer
            Dropout(0.1, seed=2022),
            Dense(num_classes, activation='softmax'),  # ouput layer
        ])
        model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.fit(
            Train,
            to_categorical(labelTrain),
            epochs=epochs,
            validation_data=(Test, to_categorical(labelTest)),
        )


TrainD, TestD, LabelTest, LabelTrain = ReadFiles()

outProcessTrain = preprocessing.LabelEncoder()
outProcessTest = preprocessing.LabelEncoder()

outProcessTest.fit(LabelTest)
LabelTest = outProcessTest.transform(LabelTest)
outProcessTrain.fit(LabelTrain)
LabelTrain = outProcessTrain.transform(LabelTrain)

TrainImagesRGB, TrainImagesGray = ReadImages(TrainD, "CNN")
TestImagesRGB, TestImagesGray = ReadImages(TestD, "CNN")

newTrainRGB = np.array(Augmentation(TrainImagesRGB[:500]))

TrainRGB = TrainImagesRGB[:len(TrainImagesRGB) * 0.7]
validateRGB = TrainImagesRGB[len(TrainImagesRGB) * 0.7:]
TrainGray = TrainImagesGray[:len(TrainImagesGray) * 0.7]
validateGray = TrainImagesGray[len(TrainImagesGray) * 0.7:]

newTrainRGBAll = newTrainRGB + TrainImagesRGB

CNNModel1(TrainImagesRGB, LabelTrain, TestImagesRGB, LabelTest, imageType="rgb")
CNNModel2(TrainImagesRGB, LabelTrain, TestImagesRGB, LabelTest, imageType="rgb")
print("########################################################################")
CNNModel1(TrainImagesGray, LabelTrain, TestImagesGray, LabelTest, imageType="gray")
CNNModel2(TrainImagesGray, LabelTrain, TestImagesGray, LabelTest, imageType="gray")

TrainImagesRGB, TrainImagesGray = ReadImages(TrainD, "VGG")
TestImagesRGB, TestImagesGray = ReadImages(TestD, "VGG")

vgg16(newTrainRGB, LabelTrain, TestImagesRGB, LabelTest, imageType="rgb")
vgg19(newTrainRGB, LabelTrain, TestImagesRGB, LabelTest, imageType="rgb")
print("################################################################")
vgg16(TrainImagesGray, LabelTrain, TestImagesGray, LabelTest, imageType="gray")
vgg19(TrainImagesGray, LabelTrain, TestImagesGray, LabelTest, imageType="gray")
print("################################################################")

res50(TrainImagesRGB, LabelTrain, TestImagesRGB, LabelTest, imageType="rgb")
res50(TrainImagesGray, LabelTrain, TestImagesGray, LabelTest, imageType="gray")
