import pandas as pd
import numpy as np
import scipy.misc as misc
from sklearn.preprocessing import OneHotEncoder

class EmotionDetector:
    def __init__(self):
        pass

    def make_onehot(self,x,num_labels=7):
        enc = OneHotEncoder(n_values=num_labels)

        return enc.fit_transform(np.array(x).reshape(-1, 1)).toarray()

    def load_training_dataset(self, pagepath, perc_validation=0.1, perc_test=0.1, image_size=48):
        data_frame = pd.read_csv(pagepath)
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0).dropna()
        df_images = np.vstack(data_frame['Pixels']).reshape(-1, image_size, image_size) #, 1)
        print(df_images)
        df_labels = self.make_onehot(data_frame['Emotion'])
        print(df_labels)


if __name__ == '__main__':
    app = EmotionDetector()
    app.load_training_dataset("train.csv")