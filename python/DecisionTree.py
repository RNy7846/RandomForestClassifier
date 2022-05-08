import CSVReader
import random
from math import log, sqrt


class DecisionTreeClassifier:

    class DecisionNode:
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.results = results
            self.tb = tb
            self.fb = fb

    def __init__(self, max_depth=-1, random_features=False):
        self.root_node = None
        self.max_depth = max_depth
        self.features_indexes = []
        self.random_features = random_features

    def fit(self, rows, criterion=None):
        if len(rows) < 1:
            raise ValueError("Not enough samples in the given dataset")

        if not criterion:
            criterion = self.entropy
        if self.random_features:
            self.features_indexes = self.choose_random_features(rows[0])
            rows = [self.get_features_subset(row) + [row[-1]] for row in rows]

        self.root_node = self.build_tree(rows, criterion, self.max_depth)


    def predict(self, features):
        if self.random_features:
            if not all(i in range(len(features))
                       for i in self.features_indexes):
                raise ValueError("The given features don't match\
                                 the training set")
            features = self.get_features_subset(features)

        return self.classify(features, self.root_node)

    def choose_random_features(self, row):
        nb_features = len(row) - 1
        return random.sample(range(nb_features), int(sqrt(nb_features)))


    def get_features_subset(self, row):
        return [row[i] for i in self.features_indexes]


    def divide_set(self, rows, column, value):
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]

        return set1, set2

    def unique_counts(self, rows):
        results = {}
        for row in rows:
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    def entropy(self, rows):
        log2 = lambda x: log(x) / log(2)
        results = self.unique_counts(rows)
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * log2(p)
        return ent

    def build_tree(self, rows, func, depth):
        if len(rows) == 0:
            return self.DecisionNode()
        if depth == 0:
            return self.DecisionNode(results=self.unique_counts(rows))

        current_score = func(rows)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(rows[0]) - 1

        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                set1, set2 = self.divide_set(rows, col, value)

                p = float(len(set1)) / len(rows)
                gain = current_score - p * func(set1) - (1 - p) * func(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        if best_gain > 0:
            trueBranch = self.build_tree(best_sets[0], func, depth - 1)
            falseBranch = self.build_tree(best_sets[1], func, depth - 1)
            return self.DecisionNode(col=best_criteria[0],
                                     value=best_criteria[1],
                                     tb=trueBranch, fb=falseBranch)
        else:
            return self.DecisionNode(results=self.unique_counts(rows))


    def classify(self, observation, tree):
        if tree.results is not None:
            return list(tree.results.keys())[0]
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)


def test_tree():
    data = CSVReader.read_csv("../data/income.csv")
    tree = DecisionTreeClassifier(random_features=True)
    tree.fit(data)

    print(tree.predict([39, 'State-gov', 'Bachelors', 13, 'Never-married',
                        'Adm-clerical', 'Not-in-family', 'White', 'Male',
                        2174, 0, 40, 'United-States']))


if __name__ == '__main__':
    test_tree()
