"""
Decision tree data structure (recursive).
Authors: Luke Slyder and Sam Lowenstein
Date: 12/15/2019
"""
import copy
from Partition import *

class DecisionTree:

    def __init__(self,dataset,mode,laplace,max_depth=2,current_depth=0):
        """
        Constructs a heavily modified version of the Decision Tree algorithm
        presented in class. This version doesn't use entropy. Instead, since
        there are only two "features," the first level of the tree looks at team
        1's picks, and the second looks at team 2's picks.

        Additionally, rather than restrict each example to a single path through
        the tree, each example takes 25 paths (one for each combination of a
        pick from team 1 and a pick from team 2) and contributes to the results
        of (or, if testing, predicts based on the results of) 25 different
        leaves.

        Notable parameters include mode, which is equal to "percent" if each
        leaf contains the percent chance of a team 1 victory, or "total" if each
        leaf contains the numbers of victories from each team; and laplace,
        which controls the Laplace count (used only if mode is "percent").
        """
        self.depth=current_depth
        self.count=[0,0]
        self.mode = mode
        if self.depth==max_depth:
            total_1 = laplace
            total_2 = laplace
            for lab in dataset.all_labels:
                if lab == 1:
                    total_1 += 1
                else:
                    total_2 += 1
            if total_1 == 0:
                if total_2 == 0:
                    percent_A = .5
                else:
                    percent_A = 0
            else:
                if total_2 == 0:
                    percent_A = 1.0
                else:
                    percent_A = total_1/(total_1+total_2)
            if self.mode == "percent":
                self.root = percent_A
            elif self.mode == "total":
                self.root = (total_1, total_2)
            else:
                print("Error: Invalid mode.")
                return
        else:
            self.root = 0
            self.children = []
            for child_num in range(519):
                if self.depth == 0:
                    print("Training:",child_num)
                new_data = []
                for ex in dataset.data:
                    if child_num in ex.features[("t"+str(self.depth+1)+"_champs")]:
                        new_data += [ex]
                new_dataset = Partition(new_data, dataset.F, 2)
                child = DecisionTree(new_dataset,self.mode,laplace,current_depth=self.depth+1)
                self.children+=[child]

    def run_one(self,example):
        """
        Predicts the label of a single example by following 25 different paths
        through the tree.
        """
        if self.depth==0:
            first_picks = example.features["t1_champs"]
            second_picks = example.features["t2_champs"]
            total_percent = 0
            overall_winner = 0
            for first_pick in first_picks:
                for second_pick in second_picks:
                    first_stop = self.children[first_pick]
                    second_stop = first_stop.children[second_pick]
                    if self.mode == "percent":
                        total_percent += second_stop.root
                    elif self.mode == "total":
                        overall_winner += second_stop.root[0] - second_stop.root[1]
                    else:
                        print("Error: Invalid mode.")
                        return
            if self.mode == "percent":
                final_percent = total_percent/25
                if final_percent > .5:
                    return 1
                else:
                    return 2
            elif self.mode == "total":
                if overall_winner > 0:
                    return 1
                else:
                    return 2
            else:
                print("Error: Invalid mode.")
                return
