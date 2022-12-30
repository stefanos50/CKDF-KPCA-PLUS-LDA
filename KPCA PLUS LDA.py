import multiprocessing
import threading
from multiprocessing import freeze_support
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.model_selection import KFold
import data_preprocessing
import kernels
import numpy as np
from numpy.linalg import matrix_rank
import matplotlib.pyplot as pp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import time


def sort_eigh(eigvals,eigvecs,n):
    indexes = eigvals.argsort()[::-1]
    return eigvecs[:, indexes[0: n]]

def construct_R_gram(type="train",train_set=[],test_set=[],kernel="linear",c=0,alpha=0,degree=0,sigma=0):
    if kernel == "rbf":
        if type=="train":
            return kernels.rbf(train_set,train_set,sigma)
        else:
            return kernels.rbf(test_set,train_set,sigma)
    elif kernel == "linear":
        if type=="train":
            return kernels.linear(train_set,train_set)
        else:
            return kernels.linear(test_set,train_set)
    elif kernel == "poly":
        if type=="train":
            return kernels.polynomial(train_set,train_set, degree)
        else:
            return kernels.polynomial(test_set,train_set, degree)
    elif kernel == "sigmoid":
        if type == "train":
            return kernels.sigmoid(train_set, train_set, alpha, c)
        else:
            return kernels.sigmoid(test_set, train_set, alpha, c)

def calculate_scatter_matrices(X_train,y_train):
    unique_labels =  np.unique(y_train)
    mean_vecs = []
    for label in range(len(unique_labels)):
        mean_vecs.append(np.mean(X_train[y_train == unique_labels[label]], axis=0))
    d = X_train.shape[1]
    S_w = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        mean_vec = mean_vec.reshape(d, 1)
        S_w += (X_train[y_train == unique_labels[i]].T - mean_vec).dot((X_train[y_train == unique_labels[i]].T - mean_vec).T) / X_train.shape[0]
    mean_overall = np.mean(X_train, axis=0)
    S_b = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == unique_labels[i], :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        S_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T) / X_train.shape[0]
    return S_w,S_b


def train_kpca(R_gram=[]):
    M = R_gram.shape[0]
    one_M = np.ones((M,M)) / M
    R = R_gram - one_M.dot(R_gram) - R_gram.dot(one_M) + one_M.dot(R_gram).dot(one_M)
    n_components = matrix_rank(R)
    eigenvalues, eigenvectors = scipy.linalg.eigh(R)
    eigenvectors_root = np.sqrt(eigenvalues)
    eigenvectors = eigenvectors / eigenvectors_root
    P = sort_eigh(eigenvalues,eigenvectors,n_components)
    kpca_result = np.dot(R_gram, P)
    return kpca_result,P

def test_kpca(R_gram=[],P_train=[]):
    kpca_result = np.dot(R_gram, P_train)
    return kpca_result

def get_class_indexes(labels):
    labels = np.array(labels)
    unique_labels =  np.unique(labels)
    idx = []
    for i in range(len(unique_labels)):
        found = np.where(labels == unique_labels[i])[0].tolist()
        idx.append(found)
    return idx

def LDA(X_train,y_train,X_test,y_test,theta):
    try:
        print("Thread "+str(threading.get_ident())+":Calculating between and within scatter matrices...")
        S_w , S_b = calculate_scatter_matrices(X_train,y_train)

        print("Thread "+str(threading.get_ident())+":Calculating regular and irregular vectors for training and test set...")
        q = matrix_rank(S_w)

        eig_vals_sw, eig_vecs_sw = scipy.linalg.eigh(S_w)

        #eig_vecs_sw = sort_eigh(eig_vals_sw,eig_vecs_sw,len(eig_vals_sw))

        P1 = []
        P2 = []
        for i in range(len(eig_vecs_sw)):
            if i < q:
                P1.append(eig_vecs_sw[i])
            else:
                P2.append(eig_vecs_sw[i])
        P1 = np.array(P1)
        P2 = np.array(P2)

        SB = np.array(P1).dot(S_b).dot(P1.transpose())
        SW = np.array(P1).dot(S_w).dot(P1.transpose())

        d = matrix_rank(SB)

        eigvals, eigvecs = scipy.linalg.eigh(SB, SW,subset_by_index=[SW.shape[0]-d,SW.shape[0]-1])

        U = np.array(eigvecs)

        regular_vector_train = np.array(U.transpose()).dot(P1).dot(X_train.transpose())
        regular_vector_test = np.array(U.transpose()).dot(P1).dot(X_test.transpose())

        SB = np.array(P2).dot(S_b).dot(P2.transpose())
        eigvals, eigvecs = eigh(SB,subset_by_index=[SB.shape[0]-d,SB.shape[0]-1])

        V = np.array(eigvecs)

        irregular_vector_train = np.array(V.transpose()).dot(P2).dot(X_train.transpose())
        irregular_vector_test = np.array(V.transpose()).dot(P2).dot(X_test.transpose())


        print("Thread "+str(threading.get_ident())+":Calculating accuracy for Nearest Neighbor and Mean distance classifiers...")
        #Calculate test accuracy using nearest neighbor classifier
        NN = nearest_neighbor_classifier(regular_vector_train,irregular_vector_train,regular_vector_test,irregular_vector_test,np.array(y_train),y_test,theta)
        print("Thread "+str(threading.get_ident())+":Test set accuracy for NN: "+str(NN*100)+"%")

        #Calculate test accuracy using minimum distance classifier
        MD = minimum_distance_classifier(regular_vector_train,irregular_vector_train,regular_vector_test,irregular_vector_test,y_train,y_test,theta)
        print("Thread "+str(threading.get_ident())+":Test set accuracy for MD: "+str(MD*100)+"%")

        return NN,MD
    except Exception as e:
        print(e)
        print("Thread "+str(threading.get_ident())+"Failed calculating in LDA...")
        print("Thread "+str(threading.get_ident())+":Test set accuracy for NN: 0%")
        print("Thread " + str(threading.get_ident()) + ":Test set accuracy for MD: 0%")
        return 0.0,0.0

def nearest_neighbor_classifier(regular_vector_train,irregular_vector_train,regular_vector_test,irregular_vector_test,real_labels,test_labels,theta):
    g = fuse(regular_vector_train.transpose(),irregular_vector_train.transpose(),regular_vector_test.transpose(),irregular_vector_test.transpose(),theta)
    m_idx = np.argmin(g, axis=0)
    predicted_labels = real_labels[m_idx]
    return calculate_classifier_accuracy(predicted_labels,test_labels)

def minimum_distance_classifier(regular_vector_train,irregular_vector_train,regular_vector_test,irregular_vector_test,train_labels,test_labels,theta):
    regular_mean = []
    irregular_mean = []
    indx = get_class_indexes(train_labels)
    for i in range(len(indx)):
        regular_mean.append(np.mean(np.take(regular_vector_train,indx[i],axis=1), axis=1).tolist())
        irregular_mean.append(np.mean(np.take(irregular_vector_train,indx[i],axis=1), axis=1).tolist())
    regular_mean = np.array(regular_mean)
    irregular_mean = np.array(irregular_mean)
    g = fuse(regular_mean,irregular_mean,regular_vector_test.transpose(),irregular_vector_test.transpose(),theta)
    m_idx = np.argmin(g, axis=0) + 1
    return calculate_classifier_accuracy(m_idx,test_labels)

def fuse(z_one_train,z_two_train,z_one_test,z_two_test,theta):
    regular_distances = cdist(z_one_train,z_one_test, metric='euclidean')
    regular_distances_sum = np.sum(regular_distances,axis=1);
    for i in range(len(regular_distances)):
            regular_distances[i] = regular_distances[i] * (1/(regular_distances_sum[i]))
    irregular_distances = cdist(z_two_train,z_two_test, metric='euclidean')
    irregular_distances_sum = np.sum(irregular_distances,axis=1);
    for i in range(len(irregular_distances)):
        irregular_distances[i] = irregular_distances[i] * (1/(irregular_distances_sum[i]))
    g =  np.add(theta * regular_distances,irregular_distances)
    return g

def calculate_classifier_accuracy(predicted,real):
    total = len(predicted)
    counter = 0
    for i in range(len(predicted)):
        if predicted[i] == real[i]:
            counter += 1
    return (counter/total)

def KPCA_PLUS_LDA(x_train,y_train,x_test,y_test,kernel="linear",c=0,alpha=0,degree=0,sigma=0,theta=0):
    print("Thread "+str(threading.get_ident())+":Calculating Kernel PCA for training and test set...")
    start_time = time.time()
    R_gram_train = construct_R_gram(type="train",train_set=x_train,test_set=x_test,kernel=kernel,c=c,alpha=alpha,degree=degree,sigma=sigma)
    R_gram_test = construct_R_gram(type="test",train_set=x_train,test_set=x_test,kernel=kernel,c=c,alpha=alpha,degree=degree,sigma=sigma)
    kpca_train, P = train_kpca(R_gram=R_gram_train)
    kpca_test = test_kpca(R_gram=R_gram_test, P_train=P)
    acc1,acc2 = LDA(kpca_train[:, ~np.isnan(kpca_train).any(axis=0)], y_train,kpca_test[:, ~np.isnan(kpca_test).any(axis=0)], y_test, theta)
    return acc1,acc2,(time.time() - start_time)

def K_Fold_Validation(data,labels,N,kernel="linear",c=0,alpha=0,degree=0,sigma=0,theta=0):
    run_times = []
    NN_accuracy = []
    MD_accuracy = []
    data_array = np.array(data)
    labels_array = np.array(labels)
    kf = KFold(n_splits=N)
    for train_index, test_index in kf.split(data):
        train_data_x, test_data_x = data_array[train_index],data_array[test_index]
        train_data_y, test_data_y = labels_array[train_index],labels_array[test_index]
        acc1,acc2,time = KPCA_PLUS_LDA(train_data_x,train_data_y,test_data_x,test_data_y,kernel=kernel,c=c,sigma=sigma,theta=theta,alpha=alpha,degree=degree)
        NN_accuracy.append(acc1)
        MD_accuracy.append(acc2)
        run_times.append(time)
    print("NN average accuracy: ", sum(NN_accuracy)/len(NN_accuracy)*100)
    print("MD average accuracy: ", sum(MD_accuracy)/len(MD_accuracy)*100)
    print("Average time: ", sum(run_times) / len(run_times))
    return sum(NN_accuracy)/len(NN_accuracy),sum(MD_accuracy)/len(MD_accuracy),sum(run_times) / len(run_times)

def plot_results(accuracy_NN,accuracy_MD,title,params,y_title,x_title):
    pp.plot(params,accuracy_NN,label='NN accuracy')
    pp.plot(params,accuracy_MD,label='MD accuracy')
    pp.xlabel(x_title)
    pp.ylabel(y_title)
    pp.title(title)
    plt.legend()
    pp.show()

def grid_search_rbf(args):
    theta = args[0]
    sigma = args[1]
    cv = args[2]
    result_acc1 = []
    result_acc2 = []
    result_time = []
    params = []
    for j in range(len(sigma)):
        if cv > 1:
            acc1, acc2, time = K_Fold_Validation(X, y, cv, kernel="rbf", sigma=sigma[j], theta=theta)
        else:
            acc1, acc2, time = KPCA_PLUS_LDA(np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test), kernel="rbf", sigma=sigma[j], theta=theta)
        result_acc1.append(acc1)
        result_acc2.append(acc2)
        result_time.append(time)
        params.append(str(theta)+"\n"+str(sigma[j]))
    plot_results(result_acc1,result_acc2,"sigma="+str(sigma)+",theta="+str(theta),params,"Accuracy","theta,sigma")
    pp.plot(params,result_time)
    pp.title("sigma="+str(sigma)+",theta="+str(theta))
    pp.show()

def grid_search_linear(args):
    theta = args[0]
    cv = args[1]
    result_acc1 = []
    result_acc2 = []
    result_time = []
    params = []
    for j in range(len(theta)):
        if cv > 1:
            acc1, acc2, time = K_Fold_Validation(X, y, cv, kernel="linear", theta=theta[j])
        else:
            acc1, acc2, time = KPCA_PLUS_LDA(np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test), kernel="linear", theta=theta[j])
        result_acc1.append(acc1)
        result_acc2.append(acc2)
        result_time.append(time)
        params.append(str(theta[j]))
    plot_results(result_acc1,result_acc2,"theta="+str(theta),params,"Accuracy","theta")
    pp.plot(params,result_time)
    pp.title("theta="+str(theta))
    pp.show()

def grid_search_sigmoid(args):
    theta = args[0]
    a = args[1]
    c = args[2]
    cv = args[3]
    result_acc1 = []
    result_acc2 = []
    result_time = []
    params = []
    for i in range(len(a)):
        for j in range(len(c)):
            if cv > 1:
                acc1, acc2, time = K_Fold_Validation(X, y, cv, kernel="sigmoid", alpha=a[i],c=c[j], theta=theta)
            else:
                acc1, acc2, time = KPCA_PLUS_LDA(np.array(x_train), np.array(y_train), np.array(x_test),
                                                 np.array(y_test), kernel="sigmoid", alpha=a[i],c=c[j], theta=theta)
            result_acc1.append(acc1)
            result_acc2.append(acc2)
            result_time.append(time)
            params.append(str(theta)+"\n"+str(a[i])+"\n"+str(c[j]))
    plot_results(result_acc1,result_acc2,"c="+str(c)+",theta="+str(theta)+",a="+str(a),params,"Accuracy","theta,a,c")
    pp.plot(params,result_time)
    pp.title("c="+str(c)+",theta="+str(theta)+",a="+str(a))
    pp.show()

def grid_search_poly(args):
    theta = args[0]
    d = args[1]
    cv = args[2]
    result_acc1 = []
    result_acc2 = []
    result_time = []
    params = []
    for k in range(len(d)):
        if cv > 1:
            acc1, acc2, time = K_Fold_Validation(X, y, cv, kernel="poly",degree=d[k], theta=theta)
        else:
            acc1, acc2, time = KPCA_PLUS_LDA(np.array(x_train), np.array(y_train), np.array(x_test),
                                                     np.array(y_test), kernel="poly",degree=d[k], theta=theta)
        result_acc1.append(acc1)
        result_acc2.append(acc2)
        result_time.append(time)
        params.append(str(theta)+"\n"+str(d[k]))
    plot_results(result_acc1,result_acc2,"theta="+str(theta)+",d="+str(d),params,"Accuracy","theta,d")
    pp.plot(params,result_time)
    pp.title("theta="+str(theta)+",d="+str(d))
    pp.show()
#eeg emotions dataset
#X,y = data_preprocessing.get_eeg_biosignals("G:\\emotions.csv")

#Olivetti dataset
X,y = data_preprocessing.get_olivetti()

#Statlog Shuttle dataset
#x_train,y_train = data_preprocessing.get_shuttle_dataset("G:\\shuttle_train..txt")
#x_test,y_test = data_preprocessing.get_shuttle_dataset("G:\\shuttle_test.txt")


if __name__ == '__main__':
    freeze_support()
    cv = 10
    theta = [0.15,0.25,0.5,0.75,1]
    c = [0.001,0.01,0.1,0.3]
    sigma = [5,6,7,8,9,10,11,12]
    a = [0.001,0.01,0.1,0.3]
    d = [1,2,3,4]
    # create a process pool that uses all cpus
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        #pool.map(grid_search_linear,[[theta,cv]])
        pool.map(grid_search_rbf,[[theta[0], sigma,cv], [theta[1], sigma,cv], [theta[2], sigma,cv], [theta[3], sigma,cv], [theta[4], sigma,cv]])
        #pool.map(grid_search_sigmoid, [[theta[0], a,c,cv], [theta[1], a,c,cv], [theta[2], a,c,cv], [theta[3], a,c,cv], [theta[4], a,c,cv]])
        #pool.map(grid_search_poly, [[theta[0],d,cv], [theta[1],d,cv], [theta[2],d,cv], [theta[3],d,cv], [theta[4],d,cv]])



