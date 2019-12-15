"""
NaiveBayes class. A Naive Bayes model that holds information taken from the
input training data and can classify testing examples. Adapted from Lab 4.
Authors: Luke Slyder and Sam Lowenstein
Date: 12/15/2019
"""
from Partition import *
import numpy as np

class NaiveBayes:

    def __init__(self, dataset):
        """
        Constructs and trains a Naive Bayes model based on dataset.
        """
        K = dataset.K
        data = dataset.data
        n = dataset.n
        num_features = dataset.num_features
        max_mag=dataset.max_mag
        F = dataset.F

        N_k = np.zeros(K)
        for label in dataset.all_labels:
            label = int(label)
            N_k[label]+=1
        theta_k = np.zeros(K)
        for k in range(K):
            theta = (N_k[label] + 1) / (n + K)
            theta_k[k] = theta
        self.theta_k=theta_k
        N_jvk = np.zeros((num_features,max_mag,K))
        for ex in data:
            for j in range(num_features):
                j_name = list(ex.features)[j]
                v_name = ex.features[j_name]
                v = F[j_name].index(v_name)
                k = ex.label
                N_jvk[j][v][k] += 1

        theta_jvk = np.zeros((num_features,max_mag,K))
        for j in range(num_features):
            j_name = list(ex.features)[j]
            mag_f_j = len(F[j_name])
            for v in range(mag_f_j):
                for k in range(K):
                    theta = (N_jvk[j][v][k] + 1) / (N_k[k] + mag_f_j)
                    theta_jvk[j][v][k] = theta
        self.theta_jvk=theta_jvk

        self.K = K
        self.F = F

    def classify(self,x):
        """
        Using the model we've trained, attempts to classify one example x.
        """
        best_k = 0
        best_prob = 0
        for k in range(self.K):
            p_yk = self.theta_k[k]
            p_ykx = p_yk
            for j_name in x:
                v_name = x[j_name]
                j = list(self.F).index(j_name)
                v = self.F[j_name].index(v_name)
                p_ykx *= self.theta_jvk[j][v][k]
            if p_ykx > best_prob:
                best_k = k
                best_prob = p_ykx
        return best_k
