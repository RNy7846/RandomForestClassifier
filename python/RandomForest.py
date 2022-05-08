import logging
import random
from concurrent.futures import ProcessPoolExecutor

import CSVReader
from DecisionTree import DecisionTreeClassifier


class RandomForestClassifier(object):

    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers


    def fit(self, data):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            rand_fts = map(lambda x: [x, random.sample(data, self.nb_samples)],
                           range(self.nb_trees))
            self.trees = list(executor.map(self.train_tree, rand_fts))


    def train_tree(self, data):
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data[1])
        return tree


    def predict(self, feature):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(feature))

        return max(set(predictions), key=predictions.count)


def test_rf():
    from sklearn.model_selection import train_test_split

    data = CSVReader.read_csv("../data/income.csv")
    train, test = train_test_split(data, test_size=0.3)

    rf = RandomForestClassifier(nb_trees=60, nb_samples=3000, max_workers=4)
    rf.fit(train)

    errors = 0
    features = [ft[:-1] for ft in test]
    values = [ft[-1] for ft in test]

    for feature, value in zip(features, values):
        prediction = rf.predict(feature)
        if prediction != value:
            errors += 1

    logging.info("Error rate: {}".format(errors / len(features) * 100))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_rf()
