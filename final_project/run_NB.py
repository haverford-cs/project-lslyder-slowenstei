"""
Runs Naive Bayes on the given dataset. Adapted from Lab 4.
Authors: Luke Slyder and Sam Lowenstein
Date: 12/15/2019
"""

import util
import math
from NaiveBayes import *
def main():
    """
    Loads data into partitions, creates a Naive Bayes model based on the train
    data, runs the model on the test data, and evaluates its accuracy.
    """
    opts = util.parse_args()
    train_partition, test_partition = util.read_arff(opts.filename)

    nb_model = NaiveBayes(train_partition)

    examples = test_partition.data
    total = len(examples)
    total_correct = 0

    K = test_partition.K
    confusion_matrix = np.zeros((K,K), int)
    for example in examples:
        y_hat = nb_model.classify(example.features)
        y = example.label
        confusion_matrix[y][y_hat] += 1

        if y_hat == y:
            total_correct += 1

    accuracy = round(total_correct/total,6)
    accuracy_str = "Accuracy: " + str(accuracy) + " ("
    correct_str = str(total_correct) + " out of " + str(total) + " correct)"
    print(accuracy_str + correct_str)
    stretch = 8
    prediction_labels = "   "
    top_row = "   "
    table = ""
    for y_hat in range(K):
        prediction_labels += " " * (stretch - len(str(y_hat+1))) + str(y_hat+1)
        top_row += "-" * stretch
    for y in range(K):
        table +=  " " + str(y+1) + "|"
        for y_hat in range(K):
            entry = str(confusion_matrix[y][y_hat])
            table += " " * (stretch - len(entry)) + entry
        table+="\n"
    print("\n\n        prediction")
    print(prediction_labels)
    print(top_row)
    print(table)



main()
