import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def load_dataset(data_file: str):
    df = pd.read_csv(data_file, sep=',', header=None)
    return df


def visualize_data(df: pd.DataFrame):
    # get wrist data(column 0-5) and visualize
    data = df.values
    plt.figure()
    plt.subplot(211)
    plt.plot(data[:, 0:3])
    plt.title('Accelerometer data of Wrist in dataset1')

    plt.subplot(212)
    plt.plot(data[:, 3:6])
    plt.title('Gyroscope data of Wrist in dataset1')
    plt.show()


def remove_noises(df: pd.DataFrame):
    # removing noises by low-pass filter
    b, a = signal.butter(4, 0.6, 'low', analog=False)
    row, col = df.shape

    data = df.values
    for i in range(col):
        data[:, i] = signal.filtfilt(b, a, data[:, i])
    return data


def extract_features():
    training = np.empty(shape=(0, 73))
    testing = np.empty(shape=(0, 73))
    # deal with each dataset file
    for i in range(19):
        df = load_dataset(f'dataset/dataset_{i + 1}.txt')
        print('deal with dataset ' + str(i + 1))
        for c in range(1, 14):
            activity_data = remove_noises(df[df[24] == c])
            datat_len = len(activity_data)
            training_len = math.floor(datat_len * 0.8)
            training_data = activity_data[:training_len, :]
            testing_data = activity_data[training_len:, :]

            # data segementation: for time series data, we need to segment the whole time series,
            # and then extract features from each period of time to represent the raw data. In
            # this example code, we define each period of time contains 1000 data points. Each
            # period of time contains different data points. You may consider overlap segmentation,
            # which means consecutive two segmentation share a part of data points, to
            # get more feature samples.
            training_sample_number = training_len // 1000 + 1
            testing_sample_number = (datat_len - training_len) // 1000 + 1

            for s in range(training_sample_number):
                if s < training_sample_number - 1:
                    sample_data = training_data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = training_data[1000*s:, :]
                # in this example code, only three accelerometer data in wrist sensor is used to
                # extract three simple features: min, max, and mean value in a period of time.
                # Finally we get 9 features and 1 label to construct feature dataset. You may
                # consider all sensors' data and extract more

                feature_sample = []
                for col in range(24):
                    feature_sample.append(np.min(sample_data[:, col]))
                    feature_sample.append(np.max(sample_data[:, col]))
                    feature_sample.append(np.mean(sample_data[:, col]))
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
                training = np.concatenate((training, feature_sample), axis=0)

            for s in range(testing_sample_number):
                if s < training_sample_number - 1:
                    sample_data = testing_data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = testing_data[1000*s:, :]

                feature_sample = []
                for col in range(24):
                    feature_sample.append(np.min(sample_data[:, col]))
                    feature_sample.append(np.max(sample_data[:, col]))
                    feature_sample.append(np.mean(sample_data[:, col]))
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
                testing = np.concatenate((testing, feature_sample), axis=0)

    df_training = pd.DataFrame(training)
    df_testing = pd.DataFrame(testing)
    # df_training.to_csv('training_data.csv', index=None, header=None)
    # df_testing.to_csv('testing_data.csv', index=None, header=None)
    return df_training, df_testing


def prepare_training_set(df_training: pd.DataFrame, df_testing: pd.DataFrame):
    _, n_features = df_training.shape

    # Labels should start from 0 in sklearn
    y_train = df_training[n_features - 1].values - 1.0
    y_train = np.array([int(item) for item in y_train])

    y_test = df_testing[n_features - 1].values - 1.0
    y_test = np.array([int(item) for item in y_test])

    df_training = df_training.drop([n_features - 1], axis=1)
    x_train = df_training.values

    df_testing = df_testing.drop([n_features - 1], axis=1)
    x_test = df_testing.values

    # Feature normalization for improving the performance of machine learning models. In this example code,
    # StandardScaler is used to scale original feature to be centered around zero. You could try other
    # normalization methods.
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def training_models(x_train, y_train, x_test, y_test):
    # Build KNN classifier, in this example code
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(x_train, y_train)

    # Evaluation. when we train a machine learning model on training set, we should evaluate its
    # performance on testing set.We could evaluate the model by different metrics. Firstly, we
    # could calculate the classification accuracy. In this example
    # code, when n_neighbors is set to 4, the accuracy achieves 0.757.
    y_pred = knn.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    # We could use confusion matrix to view the classification for each activity.
    print(confusion_matrix(y_test, y_pred))

    # Another machine learning model: svm. In this example code, we use gridsearch to find the optimial classifier
    # It will take a long time to find the optimal classifier.
    # the accuracy for SVM classifier with default parameters is 0.71,
    # which is worse than KNN. The reason may be parameters of svm classifier are not optimal.
    # Another reason may be we only use 9 features and they are not enough to build a good svm classifier.
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 100]},
        {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    ]
    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x_train, y_train)
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    data_ = load_dataset('dataset/dataset_1.txt')
    # just visualize 6 columns
    visualize_data(data_[range(6)])
    # make sitting data smooth
    smooth_data = remove_noises(data_[data_[24] == 1][list(range(24))])
    visualize_data(pd.DataFrame(smooth_data))

    df_training, df_testing = extract_features()
    train_x, train_y, test_x, test_y = prepare_training_set(df_testing=df_testing, df_training=df_training)
    training_models(train_x, train_y, test_x, test_y)
