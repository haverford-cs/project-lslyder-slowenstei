"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Authors: Sara Mathieson, Luke Slyder, and Sam Lowenstein
Date: 12/15/2019
"""

from math import *

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data, F, K):
        """Store information about a dataset"""
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.num_features = len(self.F)
        self.n = len(self.data)
        labels_list = []
        for ex in self.data:
            if ex.label not in labels_list:
                labels_list+=[ex.label]
        self.labels_list=labels_list
        all_labels=[]
        for ex in self.data:
            all_labels+=[ex.label]
        self.all_labels=all_labels
        self.K = K
        self.max_mag = 0
        for feature in self.F:
            mag = len(self.F[feature])
            self.max_mag = max(self.max_mag, mag)

# class Partition:
#
#     def __init__(self, data, F):
#         """Store information about a dataset"""
#         self.data = data # list of examples
#         # dictionary. key=feature name: value=set of possible values
#         self.F = F
#         self.n = len(self.data)
#         labels_list = []
#         for ex in self.data:
#             if ex.label not in labels_list:
#                 labels_list+=[ex.label]
#         self.labels_list=labels_list
#         all_labels=[]
#         for ex in self.data:
#             all_labels+=[ex.label]
#         self.all_labels=all_labels

    # TODO: implement entropy and information gain methods here!

    def label_probability(self,label):
        """
        Returns the probability of label being the label for a random example in
        self.data.
        """
        total = self.n
        quantity = 0
        for ex in self.data:
            if ex.label == label:
                quantity+=1
        prob = quantity/total
        return prob

    def feature_probability(self,feature,value):
        """
        Returns the probability of value being the value of feature for a random
        example in self.data.
        """
        total=self.n
        quantity=0
        for ex in self.data:
            if ex.features[feature]==value:
                quantity+=1
        prob = quantity/total
        return prob

    def probability_given_value(self,label,feature,value):
        """
        Returns the probability of label being the label for a random example in
        the subset of self.data containing exactly those examples for which
        value is the value of feature.
        """
        total=self.n
        quantity=0
        for ex in self.data:
            if ex.features[feature]==value:
                if ex.label == label:
                    quantity+=1
        prob = quantity/total
        if prob!=0:
            cond_prob=prob/self.feature_probability(feature,value)
        else:
            cond_prob=0
        return cond_prob

    def entropy(self):
        """
        Returns H(Y) for our data.
        """
        ent = 0
        for label in self.labels_list:
            prob=self.label_probability(label)
            if prob!=0:
                ent += prob*-log(prob,2)
        return ent

    def conditional_entropy(self,feature,value):
        """
        Returns H(Y|feature=value) for our data.
        """
        ent=0
        for label in self.labels_list:
            prob=self.probability_given_value(label,feature,value)
            if prob!=0:
                ent += prob*-log(prob,2)
        return ent

    def list_of_values(self,feature):
        """
        Returns the list of possible values for feature in self.data.
        """
        values=[]
        for ex in self.data:
            if ex.features[feature] not in values:
                values+=[ex.features[feature]]
        return values


    def feature_entropy(self,feature):
        """
        Returns H(Y|feature) for our data.
        """
        ent=0
        values=self.F[feature]
        for value in values:
            prob=self.feature_probability(feature,value)
            cond_ent=self.conditional_entropy(feature,value)
            ent+=prob*cond_ent
        return ent

    def information_gain(self,feature):
        """
        Returns the information gain H(Y)-H(Y|X) for our data.
        """
        total_ent=self.entropy()
        feat_ent=self.feature_entropy(feature)
        info_gain=total_ent-feat_ent
        return info_gain

    def max_information_gain(self):
        """
        Returns the ordered pair (feature for which information gain is
        maximized in our data, information gain for said feature).
        """
        maximum=("dummy",-1)
        for feature in self.F:
            info_gain=self.information_gain(feature)
            if info_gain>maximum[1]:
                maximum=(feature,info_gain)
        return maximum

    def same_label(self):
        """
        Returns True if there is no more than one unique label in our data and
        False otherwise.
        """
        return len(self.labels_list)<=1
