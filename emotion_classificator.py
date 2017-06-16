import pandas as pd
import numpy as np
import scipy.misc as misc
from sklearn.preprocessing import OneHotEncoder
import pickle

class EmotionDetector:
    def __init__(self):
        self.validation_labels = None
        self.test_labels = None
        self.train_labels = None
        self.validation_images = None
        self.test_images = None
        self.train_images = None

    def make_onehot(self,x,num_labels=7):
        enc = OneHotEncoder(n_values=num_labels)
        return enc.fit_transform(np.array(x).reshape(-1, 1)).toarray()

    def load_training_dataset(
            self,
            filename_orig,
            perc_validation=0.1,
            perc_test=0.1,
            image_size=48,
            save_pickle=True,
            filename_pickle="datasets_train_valid_test.pickle"
    ):
        data_frame = pd.read_csv(filename_orig)
        data_frame['Pixels'] = (
            data_frame['Pixels']
                .apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
                .dropna())

        df_images = np.vstack(data_frame['Pixels']).reshape(-1, image_size, image_size) #, 1)
        print(df_images)

        df_labels = self.make_onehot(data_frame['Emotion'])
        print(df_labels)

        shuffle = np.random.permutation(df_images.shape[0])
        df_images = df_images[shuffle]
        df_labels = df_labels[shuffle]

        marker_validation = int(df_images.shape[0] * perc_validation)
        marker_test = marker_validation + int(df_images.shape[0] * perc_test)

        self.validation_labels = df_labels[:marker_validation]
        self.test_labels = df_labels[marker_validation:marker_test]
        self.train_labels = df_labels[marker_test:]
        self.validation_images = df_images[:marker_validation]
        self.test_images = df_images[marker_validation:marker_test]
        self.train_images = df_images[marker_test:]

        if save_pickle:
            with open(filename_pickle, "wb") as file:
                save = {
                    "validation_labels": self.validation_labels,
                    "test_labels": self.test_labels,
                    "train_labels": self.train_labels,
                    "validation_images": self.validation_images,
                    "test_images": self.test_images,
                    "train_images": self.train_images
                }
                pickle.dump(save, file)




if __name__ == '__main__':
    app = EmotionDetector()
    app.load_training_dataset("train.csv")