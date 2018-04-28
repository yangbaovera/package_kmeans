import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import collections 
import time
import numba
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from numba import jit


#Kmeans
def distance(X, Y):
    ''' 
        Function to return distance matrix between every observation to each centroids   
        Input: X, observations of data. Y, centriod from clustering
        Output: dist, distance matrix. 
    '''
    dist = np.empty((X.shape[0], Y.shape[0]), dtype=np.float)
    for k in range(X.shape[0]):
        for i in range(Y.shape[0]):
            temp = 0
            for j in range(X.shape[1]):
                temp = temp +(X[k,j] - Y[i,j])**2
            dist[k, i]= temp**0.5
    return dist


def kmeans(data, k, centroids, max_iter=10000):
    ''' 
        Function to return final centers for the using k-means clustering algorithm and 
        jit_distance function to calculate distance matrix
        
        Input: data, an array of data. k, the number of clusters. centroids, initial centroids
        Output: C, an array with length k of initial cluster centers. 
    '''
    for i in range(max_iter):
        cdist = (distance(data, centroids))**2
        labels = np.argmin(cdist, axis=1)
        update_centroids = np.zeros(centroids.shape)
        for j in range(k):
            # check if the centroid is the closest to some data point
            if sum(labels == j) != 0:
                update_centroids[j] = np.mean(data[labels ==j], axis=0)
            else:
                # if not, leave the lone centroid unmoved
                update_centroids[j] = centroids[j]
                
        if np.allclose(update_centroids, centroids):
            print("Algorithm converged after", i, "iterations.")
            return ({"Centroids": centroids, 
                      "Labels": labels})
        else:
            centroids = update_centroids
        
    print("Warning: maximum number of iterations reached. Failed to converge.")
    
    return({"Centroids": centroids, 
            "Labels": labels})


#kmeans plus plus
def kmeans_pp(data, k, weights):
    ''' 
        Function to return final centers for the using k-means++ clustering algorithm
        Input: data, an array of data. k, the number of clusters. weights, weight for each centroid
        Output: C, an array with length k of initial cluster centers. 
    '''
    first_random = np.random.choice(data.shape[0], 1)
    C = data[first_random, :]
    
    for i in range(k-1):
        cdist = (distance(data, C))**2
        cdist_min = np.min(cdist, axis = 1)* weights
        prob = cdist_min/np.sum(cdist_min)
        new_center = np.random.choice(data.shape[0],1, p=prob)
        C = np.vstack([C, data[new_center,:]])
        
    return C


#kmeansII
def get_weight(C, data): 
    ''' 
        Function to return  weight for each centorid
        Input: data, an array of data. centroids, initial centroids
        Output: C, an array with length k of weight for cluster centers. 
    '''
    weights=np.zeros(C.shape[0])
    cdist = (distance(data,C))**2
    min_cdist = np.argmin(cdist, axis = 1)
    count = collections.Counter(min_cdist) 
    weights = list(collections.OrderedDict(sorted(count.items(), key=lambda x: x[0])).values())
    weights=np.array(weights)/sum(weights)
    return weights



def kmeans_II(data, k, l, max_iter=10000):
    ''' 
        Function to return final centers for the using k-means|| clustering algorithm
        Input: data, an array of data. k, the number of clusters. l, oversampling factor
        Output: C, an array with length k of initial cluster centers. 
    '''
    first_random = np.random.choice(data.shape[0], 1)
    C = data[first_random, :]
    
    cdist = (distance(data, C))**2
    cdist_min = np.min(cdist, axis = 1)
    cost_phi = np.sum(cdist_min)
    
    for i in range(int(round(np.log(cost_phi)))):
        cdist = (distance(data, C))**2
        cdist_min = np.min(cdist, axis = 1)
        prob = cdist_min * l/np.sum(cdist_min)
        for j in range(data.shape[0]):
            if np.random.uniform() <= prob[j] and data[j,:] not in C:
                C = np.vstack([C, data[j,:]])
   
    weights= get_weight(C, data)

    return kmeans_pp(C, k, weights)
    

#Cdist Version

def cdist_kmeans(data, k, centroids, max_iter=10000):
    ''' 
        Function to return final centers for the using k-means clustering algorithm and 
        cdist function to calculate distance matrix
        
        Input: data, an array of data. k, the number of clusters. centroids, initial centroids
        Output: C, an array with length k of initial cluster centers. 
    '''
    
    for i in range(max_iter):
        cdist = (dist.cdist(data, centroids))**2
        labels = np.argmin(cdist, axis=1)
        update_centroids = np.zeros(centroids.shape)
        for j in range(k):
            # check if the centroid is the closest to some data point
            if sum(labels == j) != 0:
                update_centroids[j] = np.mean(data[labels ==j], axis=0)
            else:
                # if not, leave the lone centroid unmoved
                update_centroids[j] = centroids[j]
                
        if np.allclose(update_centroids, centroids):
            print("Algorithm converged after", i, "iterations.")
            return ({"Centroids": centroids, 
                      "Labels": labels})
        else:
            centroids = update_centroids
        
    print("Warning: maximum number of iterations reached. Failed to converge.")
    return centroids


def cdist_kmeans_pp(data, k, weights):
    ''' 
        Function to return final centers for the using k-means++ clustering algorithm
        Input: data, an array of data. k, the number of clusters. weights, weight for each initial centroids
        Output: C, an array with length k of initial cluster centers. 
    '''
    first_random = np.random.choice(data.shape[0], 1)
    C = data[first_random, :]
    
    for i in range(k-1):
        cdist = (dist.cdist(data, C))**2
        cdist_min = np.min(cdist, axis = 1)* weights
        prob = cdist_min/np.sum(cdist_min)
        new_center = np.random.choice(data.shape[0],1, p=prob)
        C = np.vstack([C, data[new_center,:]])
        
    return C

def cdist_get_weight(C, data):
    ''' 
        Function to return  weight for each centorid
        Input: data, an array of data. centroids, initial centroids
        Output: C, an array with length k of weight for cluster centers. 
    '''
    weights=np.zeros(C.shape[0])
    cdist = (dist.cdist(data,C))**2
    min_cdist = np.argmin(cdist, axis = 1)
    count = collections.Counter(min_cdist) 
    weights = list(collections.OrderedDict(sorted(count.items(), key=lambda x: x[0])).values())
    weights=np.array(weights)/sum(weights)
    return weights



def cdist_kmeans_II(data, k, l, max_iter=10000):
    ''' 
        Function to return final centers for the using k-means|| clustering algorithm
        Input: data, an array of data. k, the number of clusters. l, oversampling factor
        Output: C, an array with length k of initial cluster centers. 
    '''
    first_random = np.random.choice(data.shape[0], 1)
    C = data[first_random, :]
    
    cdist = (dist.cdist(data, C))**2
    cdist_min = np.min(cdist, axis = 1)
    cost_phi = np.sum(cdist_min)
    
    for i in range(int(round(np.log(cost_phi)))):
        cdist = (dist.cdist(data, C))**2
        cdist_min = np.min(cdist, axis = 1)
        prob = cdist_min * l/np.sum(cdist_min)
        for j in range(data.shape[0]):
            if np.random.uniform() <= prob[j] and data[j,:] not in C:
                C = np.vstack([C, data[j,:]])
   
    weights= cdist_get_weight(C, data)

    return cdist_kmeans_pp(C, k, weights)

#parallel Multi core
def min_distance(d, centroids):
    """
    function return the minimum distance from point d to nearest centroids
    """
    dist = np.min(np.sum((centroids - d)**2, axis=1))
    return dist

def cost_p(data, centroids): 
    """
    function that return the cost(distance) for each observation
    """

    with mp.Pool(processes = mp.cpu_count()) as pool:
        partial_dist = partial(min_distance, centroids = centroids)
        min_dist = pool.map(partial_dist, data)
        p = min_dist/np.sum(min_dist)
    return p


def random_choice(x, a, p):
    """
    helper function like np.random.choice
    but have one less argument and shift order of arguments for future map
    """
    np.random.seed()
    return np.random.choice(a = a, size = x , p =p)


def sample_p(data, distribution, l):
    
    """ 
    Function to sample l number new centers
    """  
    
    with mp.Pool(processes = mp.cpu_count()) as pool:
        partial_rc = partial(random_choice, a = len(distribution), p=distribution)
        #create l number of size one observation
        index = pool.map(partial_rc,np.repeat(1,l))
    return np.squeeze(data[index,:],axis=(1,))


def min_index_p(d, centroids):
    
    """ 
    Return the index of the minimum distance from point d 
    to its nearest centroids.
    """
    
    index = np.argmin(np.sum((centroids - d)**2, axis=1))
    return index 

def get_weight_p(data, centroids):
    
    ''' 
        Function to return  weight for each centorid
        Input: data, an array of data. centroids, initial centroids
        Output: C, an array with length k of weight for cluster centers. 
    '''

    with mp.Pool(processes = mp.cpu_count()) as pool:
        partial_min = partial(min_index_p, centroids = centroids )
        min_index = pool.map(partial_min, data)
        count = np.array([np.sum(np.array(min_index) == i) for i in range(centroids.shape[0])])
    return count/np.sum(count)



def cdist_kmeans_pp(data, k, weights):
    ''' 
        Function to return final centers for the using k-means++ clustering algorithm
        Input: data, an array of data. k, the number of clusters. weights, weight for each initial centroids
        Output: C, an array with length k of initial cluster centers. 
    '''
    first_random = np.random.choice(data.shape[0], 1)
    C = data[first_random, :]
    
    for i in range(k-1):
        cdist = (dist.cdist(data, C))**2
        cdist_min = np.min(cdist, axis = 1)* weights
        prob = cdist_min/np.sum(cdist_min)
        new_center = np.random.choice(data.shape[0],1, p=prob)
        C = np.vstack([C, data[new_center,:]])
        
    return C

def kmeans_II_p(data, k, l):
    ''' 
        Function to return final centers for the using k-means|| clustering algorithm
        Input: data, an array of data. k, the number of clusters. l, oversampling factor
        Output: C, an array with length k of initial cluster centers. 
    '''
    
    C = data[np.random.choice(range(data.shape[0]),1), :]  
    cdist = (dist.cdist(data, C))**2
    cdist_min = np.min(cdist, axis = 1)
    cost_phi = np.sum(cdist_min)
    
    for i in range(int(round(np.log(cost_phi)))):   
        
        # Calculate the cost and new distribution
        p = cost_p(data, C)
        
        # sample new centers
        C = np.r_[C, sample_p(data, p, l)]
        
    weights = get_weight_p(data,C)
    
    centers = cdist_kmeans_pp(C, k, weights)
    
    return centers


#function that return a pandas data frame with different optimization for different algorithms. 
def time_compare(data, k, l):
    
    
    kmeans_pp_time = timer(kmeans_pp, data, k, 1)
    jit_kmeans_pp_time = timer(jit_kmeans_pp, data, k, 1)
    cdist_kmeans_pp_time = timer(cdist_kmeans_pp, data, k, 1)
    ones = np.ones(data.shape[0])
    cython_kmeans_pp_time = timer(kmeans_pp_cython, ones, data, k)
    parallel_kmeans_pp_time = float('NaN')
    
    kmeans_II_time = timer(kmeans_II, data, k, l, max_iter = 10000)
    jit_kmeans_II_time = timer(jit_kmeans_II, data, k, l, max_iter = 10000)
    cdist_kmeans_II_time = timer(cdist_kmeans_II, data, k, l, max_iter = 10000)
    cython_kmeans_II_time = timer(kmeans_II_cython, data, k, l)
    parallel_kmeans_II_time = timer(kmeans_II_p,data, k, l)
        
    return pd.DataFrame([[kmeans_pp_time, kmeans_II_time],
                         [jit_kmeans_pp_time, jit_kmeans_II_time],
                         [cdist_kmeans_pp_time, cdist_kmeans_II_time],
                         [cython_kmeans_pp_time, cython_kmeans_II_time],
                         [parallel_kmeans_pp_time, parallel_kmeans_II_time]], 
                         index = ["Original","JIT", "cdist", "cython", "parallel"],
                         columns = ["k-means++","k-meansII"])



def generate_data(k, n, feature_n, var = 10):
    
    #sample k centers 
    k_centers = np.random.multivariate_normal(np.zeros(feature_n),  np.eye(feature_n)*var, k)
    
    step=round(n/k)
    points=np.ones([step*k,15])
    
    for i in range(k):
        newpoints = np.random.multivariate_normal(k_centers[i],np.eye(feature_n),size = round(n/k))
        points[i*(step):(i*(step)+(step)),:] = newpoints
        if i == 0:
            true_labels = np.repeat(i,int(n/k+n%k))
        else: 
            true_labels = np.append(true_labels,np.repeat(i,int(n/k)))
    
    points=np.append(points, k_centers,axis = 0)
    #np.random.shuffle(points)
    return (points, k_centers, true_labels)

    


def test_cluster(ini_cen, fin_cen, label, data, title): 
    plt.scatter(data[:, 0], data[:, 1], c = label)
    for i in range(np.unique(label).size):
        if i > 0:
            plt.scatter(ini_cen[i,0],ini_cen[i,1],marker = '+', c = 'red', s= 100)
            plt.scatter(fin_cen[i,0],fin_cen[i,1],marker = '*', c = 'blue', s= 100)
        else:
            plt.scatter(ini_cen[i,0],ini_cen[i,1],marker = '+', c = 'red', s= 100, label = "Initial Cluster")
            plt.scatter(fin_cen[i,0],fin_cen[i,1],marker = '*', c = 'blue', s= 100, label = "Final Cluster")
    
    plt.legend(loc='best', prop={'size':15})
    plt.title(title)