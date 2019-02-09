import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels=None, batch_size=1250, dim=(10534,), n_channels=1,
                 n_classes=10534, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X_questions = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X_questions[i,] = np.load('dataBow/qid_' + str(ID) + '.npz')['a']

            # Store class
            y[i] = np.load('dataBow/lid_' + str(ID) + '.npz')['a']

        return X_questions, y
