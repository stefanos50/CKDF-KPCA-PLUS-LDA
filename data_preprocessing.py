from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import imutils
from matplotlib import pyplot as plt
import cv2


#read the dataset from the file name
def load_dataset(filename):
    dataset = []
    with open(filename) as file:
        for line in file:
            dataset.append((line.rstrip()).split(" "))
    return dataset

def load_dataset_csv(filename,column_names):
    df = pd.read_csv(filename,names=column_names)
    return df

#remove the index (number) that indicates each column for each row
def remove_column_indx(data):
    ndataset = []
    for i in range(len(data)):
        row = []
        row.append(int(data[i][0]))
        for j in range(1,len(data[i])):
            row.append(float((data[i][j].split(":"))[1]))
        ndataset.append(row)
    return ndataset

#clear the row if a feature is missing
def clear_row_with_missing_values(data,num_features):
    ndataset = []
    for i in range(len(data)):
        if(len(data[i]) < num_features + 1):
            continue
        ndataset.append(data[i])
    return ndataset

#remove duplicated rows of the dataset
def dublicated(tdf):
    print("Dataframe shape", tdf.shape)
    tdf.drop_duplicates(subset=None, keep="first", inplace=True)
    print("Removed dublicates",tdf.shape)
    return tdf

#remove rows containing NaN-null values
def empty_cell_lines(tdf):
    print(tdf.isnull().sum(axis=1))
    return tdf.isnull().sum(axis=1).values.sum()
#calculate and print the number of rows per class
def num_class_labels(data):
    print(data['label'].value_counts())
#calculate the max and min value per column and max and min value in the dataset
def data_value_range(tdf):
    print(tdf.max(axis=0))
    print(tdf.min(axis=0))
    return min(tdf.min(axis=0)),max(tdf.max(axis=0))

#PCA technique to perform dimensionality reduction on the dataset
def PCA_reduction(tdf,n,l,o):
    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(tdf[l])
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=o)
    finalDf = pd.concat([principalDf, tdf[['label']]], axis=1)
    print("explained variance ratio:", pca.explained_variance_ratio_)
    print("Preserved Variance:", sum(pca.explained_variance_ratio_))
    return finalDf

def plot_signals(df):
    fig, axes = plt.subplots(nrows=3, ncols=1, dpi=200, figsize=(24, 12))

    df.loc[5, 'fft_0_b':'fft_749_b'].plot(title='\"fft_0_b\" Through \"fft_749_b\" -- 5', color='tab:blue', ax=axes[0])
    df.loc[15, 'fft_0_b':'fft_749_b'].plot(title='\"fft_0_b\" Through \"fft_749_b\" -- 15', color='tab:red', ax=axes[1])
    df.loc[32, 'fft_0_b':'fft_749_b'].plot(title='\"fft_0_b\" Through \"fft_749_b\" -- 32', color='tab:green', ax=axes[2])

    plt.subplots_adjust(left=0.1, bottom=0.1,
                        right=0.9, top=0.9,
                        wspace=0.4, hspace=0.4)
    plt.show()

#get iris dataset -> https://www.kaggle.com/datasets/uciml/iris
def get_iris_dataset(file_name):
    dataset = load_dataset_csv(file_name,['1','2','3','4','label'])
    dataset[dataset.columns[-1]].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],
                            [1,2,3], inplace=True)
    print("The shape of the data", dataset.shape)
    print("Lines with empty cell(s): " + str(empty_cell_lines(dataset)))
    print("False labels:", num_class_labels(dataset))
    minn, maxx = data_value_range(dataset)
    print("Min data value:", minn)
    print("Max data value:", maxx)
    scaler = StandardScaler()
    dataset[['1', '2', '3', '4']] = scaler.fit_transform(dataset[['1', '2', '3', '4']])
    dataset = PCA_reduction(dataset,2,['1', '2', '3', '4'],['principal component 1', 'principal component 2'])
    dataset = dataset.sample(frac = 1)
    y = dataset['label'].values.tolist()
    x = dataset[['principal component 1', 'principal component 2']].values.tolist()
    return x,y

#get Statlog (Shuttle) dataset -> https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#shuttle
def get_shuttle_dataset(file_name):
    dataset = load_dataset(file_name)
    dataset = remove_column_indx(dataset)
    dataset = clear_row_with_missing_values(dataset,9)
    #dataset = shuffle(dataset)
    print(dataset)
    df = pd.DataFrame(dataset)
    df = dublicated(df)
    minn, maxx = data_value_range(df)
    print("Min data value:", minn)
    print("Max data value:", maxx)
    print(df)
    y = df.iloc[:, 0].values.tolist()
    x = df.iloc[: , 1:].values.tolist()
    y_np = np.array(y)
    x_np = np.array(x)
    subset = int(x_np[y_np == 1,:].shape[0] * 0.10)
    return x_np[0:subset,:].tolist(),y_np[0:subset].tolist()

#get eeg emotions dataset -> https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
def get_eeg_biosignals(file_name):
    dataset = pd.read_csv(file_name)
    dataset = dataset.iloc[1:, :]
    dataset[dataset.columns[-1]].replace(['NEUTRAL','NEGATIVE','POSITIVE'],
                            [1,2,3], inplace=True)
    y_df = dataset.iloc[: , -1]
    x_df = dataset.iloc[: , :-1]
    scaler = StandardScaler()
    x_df = scaler.fit_transform(x_df)
    x_df = pd.DataFrame(x_df)
    y = y_df.values.tolist()
    x = x_df.values.tolist()
    return x,y

def plot_images(data):
    data = np.array(data)
    amount = len(data)
    fig = plt.figure(figsize=(12, 12))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.reshape(data[random.randint(0, amount-1)],(64,64)))
    plt.show()

def get_olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    X = faces.data
    y = faces.target + 1
    #plot_images(X)
    return X,y
