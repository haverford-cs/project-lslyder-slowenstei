"""
Runs decision tree on the given dataset. Adapted from Lab 2.
Authors: Luke Slyder and Sam Lowenstein
Date: 12/15/2019
"""

import util
import math
from DecisionTree import *
def main():
    """
    Loads data into partitions, creates a decision tree based on the train data,
    runs the decision tree on the test data, and evaluates its accuracy.
    """
    opts = util.parse_args()
    train_partition, test_partition = util.read_arff_alt(opts.filename)
    dtree=DecisionTree(train_partition, opts.mode, laplace = opts.laplace)
    print("Done with training!")
    examples=test_partition.data
    total=len(examples)
    total_correct=0
    for ex in examples:
        guess=dtree.run_one(ex)
        if guess==ex.label:
            total_correct+=1
    accuracy=round(total_correct/total,4)
    print(str(total_correct)+" out of "+str(total)+" correct")
    accuracy=str(accuracy)
    while len(accuracy)<6:
        accuracy+="0"
    print("accuracy = "+str(accuracy))

main()
