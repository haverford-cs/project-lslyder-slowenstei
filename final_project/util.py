"""
Utils for Naive Bayes and decision trees. Adapted from Labs 2 and 4.
Authors: Sara Mathieson, Luke Slyder, and Sam Lowenstein
Date: 12/15/19
"""

# python imports
from collections import OrderedDict
import math
import numpy as np
import optparse
import sys
import copy
import random

# my file imports
from Partition import *

def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-d', '--filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-m', '--mode', type='string', help='percent or total')
    parser.add_option('-l', '--laplace', type='int', help='laplace count')
    (opts, args) = parser.parse_args()

    mandatories = ['filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()
    if not opts.__dict__['mode']:
        opts.__dict__['mode'] = 'total'
    if not opts.__dict__['laplace']:
        opts.__dict__['laplace'] = 10000

    return opts

def read_arff(filename):
    """
    Read arff file into Partition format for Naive Bayes. Params:
    * filename (str), the path to the arff file
    """
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # key: feature name, value: list of feature values

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":

        clean = line.replace('{','').replace('}','').replace(',','')
        tokens = clean.split()
        name = tokens[1][1:-1]

        if '{' in line:
            feature_values = tokens[2:]
        else:
            feature_values = "cont"

        # record features or label
        F[name] = feature_values
        line = arff_file.readline().strip()

    # read the examples
    ex = 0;
    for line in arff_file:
        ex += 1;
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            if F[key] == "cont":
                val = float(tokens[i])
            X_dict[key] = val
            i += 1

        label = int(tokens[4])-1
        # add to list of Examples
        data.append(Example(X_dict,label))

    arff_file.close()

    F_disc = OrderedDict()
    f = 0
    convert_champs(data,F_disc)
    data_total = copy.deepcopy(data)
    random.Random(42).shuffle(data_total)
    data_cutoff = int(round(0.8*len(data)))
    data_train = data_total[:data_cutoff]
    data_test = data_total[data_cutoff:]
    train_partition = Partition(data_train, F_disc, 2)
    test_partition = Partition(data_test, F_disc, 2)
    return train_partition, test_partition

def read_arff_alt(filename):
    """
    Read arff file into Partition format for decision tree. Params:
    * filename (str), the path to the arff file
    """
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # key: feature name, value: list of feature values

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":

        clean = line.replace('{','').replace('}','').replace(',','')
        tokens = clean.split()
        name = tokens[1][1:-1]

        if '{' in line:
            feature_values = tokens[2:]
        else:
            feature_values = "cont"

        # record features or label
        F[name] = feature_values
        line = arff_file.readline().strip()

    # read the examples
    ex = 0;
    for line in arff_file:
        ex += 1;
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            if F[key] == "cont":
                val = float(tokens[i])
            X_dict[key] = val
            i += 1

        label = int(tokens[4])

        # add to list of Examples
        data.append(Example(X_dict,label))

    arff_file.close()

    F_disc = OrderedDict()
    f = 0
    convert_champs_alt(data,F_disc)
    data_total = copy.deepcopy(data)
    random.Random(42).shuffle(data_total)
    data_cutoff = int(round(0.8*len(data)))
    data_train = data_total[:data_cutoff]
    data_test = data_total[data_cutoff:]
    train_partition = Partition(data_train, F_disc, 2)
    test_partition = Partition(data_test, F_disc, 2)
    return train_partition, test_partition

def convert_champs(data, F_disc):
    """
    Convert champion selection features to one feature for each champion.
    Credit: based on original code from Ameet Soni.
    """

    unique = range(519)

    unique_str = []
    for un in range(519):
        unique_str += [str(un)]
    for u in unique:
        key = str(u)
        for i in range(len(data)):
            data[i].features[key] = 0
            for ind in ["t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id"]:

                if data[i].features[ind] == key:
                    data[i].features[key] = 1
            for ind in ["t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id","t2_champ5id"]:
                if data[i].features[ind] == key:
                    data[i].features[key] = 2
        F_disc[key] = [0,1,2]
    for i in range(len(data)):
        feats = copy.deepcopy(data[i].features)
        for key in feats:
            if key not in unique_str:
                del data[i].features[key]

def convert_champs_alt(data, F_disc):
    """
    Convert champion selection features to one list "feature" for each team.
    Credit: based on original code from Ameet Soni.
    """

    unique = range(519)

    for i in range(len(data)):
        data[i].features["t1_champs"] = []
        for ind in ["t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id"]:
            data[i].features["t1_champs"] += [int(data[i].features[ind])]
        data[i].features["t2_champs"] = []
        for ind in ["t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id","t2_champ5id"]:
            data[i].features["t2_champs"] += [int(data[i].features[ind])]
        feats = copy.deepcopy(data[i].features)
        for key in feats:
            if key != "t1_champs" and key != "t2_champs":
                del data[i].features[key]
    F_disc["t1_champs"] = unique
    F_disc["t2_champs"] = unique
