from keras import regularizers

import utils
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50

if __name__ == '__main__':

    name_list = ['Alekhine', 'Anand', 'Botvinnik', 'Capablanca', 'Carlsen', 'Caruana', 'Fischer', 'Kasparov', 'Morphy',
                 'Nakamura', 'Polgar', 'Tal']

    print(tf.config.list_physical_devices('GPU'))
    vec_list = utils.import_data()
    # print(utils.convert_string_to_nums_list(vec_list[0]))
    white_wins = []
    black_wins = []
    for vec in vec_list:
        # convert white pieces to black and vice versa
        black_vec = utils.convert_to_black(vec)
        black_vec = utils.convert_string_to_nums_list(black_vec)
        black_vec = utils.clean_and_convert(black_vec)
        black_wins.append(black_vec)
        black_wins.append(utils.rotate_board(black_vec))
        # set up white wins board
        vec = utils.convert_string_to_nums_list(vec)
        vec = utils.clean_and_convert(vec)
        white_wins.append(vec)
        white_wins.append(utils.rotate_board(vec))
    combined = white_wins + black_wins
    # print(white_wins[0])
    # print(len(white_wins[0]))
    # print(len(black_wins[0]))
    # print(len(utils.clean_and_convert(black_wins[0])))
    # print(white_wins[1])
    data = pd.DataFrame(combined)
    data['labels'] = np.concatenate((np.ones(len(white_wins)), np.zeros(len(black_wins))))
    new_index = []
    for i in range(len(white_wins) - 1):
        new_index.append(i)
        new_index.append(i + 1)
        new_index.append(len(white_wins) + i)
        new_index.append(len(white_wins) + i + 1)
    data = data.reindex(new_index)
    # shuffle keeping every 4 rows together
    m = data.shape[0] // 4
    data = np.array(data)
    a3D = data.reshape(m, 4, -1)
    data = pd.DataFrame(a3D[np.random.permutation(m)].reshape(-1, a3D.shape[-1]))
    # data.head()
    # print(data)
    # data = data.sample(frac=1).reset_index(drop=True)
    # print(data)
    # print(data.shape)
    # print(len(data)-len(data.drop_duplicates()))

    cutoff = int(data.shape[0] * 0.8)
    train_x = data.iloc[0:cutoff, 0:64]
    train_y = data.iloc[0:cutoff:, 64]
    test_x = data.iloc[cutoff + 1:, 0:64]
    test_y = data.iloc[cutoff + 1:, 64]

    master_list = []

    # for name in name_list:
    #     print(name)
    #     for i in range(1, 21):
    #         board_list = utils.generate_boards(i, name)
    #         #print(board_list)
    #         for board in board_list:
    #             master_list.append(board)

    # val = np.array(master_list)
    # print(val.shape)
    # val = pd.DataFrame(val)

    # val.to_csv('csv_data/data.csv')

    val = pd.read_csv('csv_data/data.csv')
    val = val.sample(frac=1).reset_index(drop=True)

    print("val shape:")
    print(val.shape)

    cutoff = int(val.shape[0] * 0.8)
    new_train_x = val.iloc[1:cutoff, 1:65]
    new_train_y = val.iloc[1:cutoff:, 65]
    new_test_x = val.iloc[cutoff + 1:, 1:65]
    new_test_y = val.iloc[cutoff + 1:, 65]

    new_train_x = np.array(new_train_x)
    new_test_x = np.array(new_test_x)
    print(new_train_x.shape)

    #reshaping for CNN
    # reshaped_train_x = []
    #
    # for i in range(new_train_x.shape[0]):
    #     reshaped_train_x.append(np.reshape(new_train_x[i], (8, 8)))
    # new_train_x = np.array(reshaped_train_x)
    #
    # reshaped_test_x = []
    # for i in range(new_test_x.shape[0]):
    #     reshaped_test_x.append(np.reshape(new_test_x[i], (8, 8)))
    # new_test_x = np.array(reshaped_test_x)
    #
    # new_train_x = new_train_x.astype(float)
    # new_test_x = new_test_x.astype(float)
    #
    # new_train_x = new_train_x[:, np.newaxis, :]
    # new_test_x = new_test_x[:, np.newaxis, :]

    print(new_train_x.shape)
    print(new_test_x.shape)
    print(new_train_y.shape)
    print(new_train_y)
    print(new_train_x)
    print(new_test_y.sum())

    # nn stuff begins
    weight_decay = 1e-4
    model = Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        # layers.Dropout(0.9),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        # layers.Dropout(0.8),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        #layers.Dropout(0.7),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        #layers.Dropout(0.6),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        # layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        layers.Dense(1, activation='sigmoid')
    ])

    # weight_decay = 1e-4
    # model = Sequential()
    # layers.Conv2D(32, 3, activation='relu', input_shape=(1, 8, 8), data_format='channels_first', kernel_regularizer=regularizers.l2(weight_decay))
    # model.add(layers.Activation('elu'))
    # model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(layers.Activation('elu'))
    # model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(layers.Activation('elu'))
    # model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(layers.Activation('elu'))
    # #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.2))
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'),)
    # model.add(layers.Dense(128, activation='relu'), )
    # model.add(layers.Dense(256, activation='relu'), )
    # model.add(layers.Dense(128, activation='relu'), )
    # model.add(layers.Dense(64, activation='relu'), )
    # model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=SGD(learning_rate=0.002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # model.fit(x=train_x, y=train_y, batch_size=128, epochs=10, verbose=1, validation_data=(test_x, test_y))
    model.fit(x=new_train_x, y=new_train_y, batch_size=128, epochs=200, verbose=1,
              validation_data=(new_test_x, new_test_y))
    model.save('model_weights')
