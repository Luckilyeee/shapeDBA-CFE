from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets
data_loader = UCR_UEA_datasets()

class PlaneDataset():
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    def load_train_data(self):
        train_data, train_label, test_data, test_label = data_loader.load_dataset("Plane")
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
        print("load train data: ", train_data.shape)
        encoder = LabelEncoder()
        self.train_label = encoder.fit_transform(train_label)
        self.train_data = train_data

        return self.train_data, self.train_label

    def load_test_data(self):
        train_data, train_label, test_data, test_label = data_loader.load_dataset("Plane")
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
        encoder = LabelEncoder()
        self.test_label = encoder.fit_transform(test_label)
        self.test_data = test_data
        return self.test_data, self.test_label


