from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class TrainModel:

    def __init__(self, config):
        self.test_size = config['split_dataset']['test_size']
        self.random_state = config['split_dataset']['random_state']


    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, self.test_size, self.random_state)
        return X_train, X_test, y_train, y_test

    def training_models(self, X, y):
        models = [DecisionTreeClassifier(), RandomForestClassifier()]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42)

        print(f'Feito o Split dos dados: test_size = {self.test_size}, random_state = {self.random_state}')
        for clf in models:
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(f'Acc:{metrics.accuracy_score(y_test, y_pred)}, do modelo, {clf}')
