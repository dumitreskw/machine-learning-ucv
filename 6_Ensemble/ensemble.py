import time
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Printer(object):
    @staticmethod
    def display(y_test, y_pred, clf, start_time):
        print("\nThe " + clf.__class__.__name__ + " took: %s seconds." % (time.time() - start_time))
        accuracy = accuracy_score(y_test, y_pred) * 100
        print("Estimation of model accuracy with " + clf.__class__.__name__ + " is ", accuracy, '%')
        print("\n")
        return accuracy

class Parser(Printer):
    def __init__(self):
        pass

    @staticmethod
    def parse_method():
        try:
            mnist = pd.read_csv('data.csv')
            X, y = mnist.drop('label', axis=1).to_numpy(), mnist['label'].to_numpy()
            X_train, X_test, y_train, y_test = X[:5000], X[5000:10000], y[:5000], y[5000:10000]

            log_clf = LogisticRegression(max_iter=10000)  
            rnd_clf = RandomForestClassifier()
            bag_clf = BaggingClassifier(RandomForestClassifier(random_state=1), n_estimators=3)

            accuracies = []

            for clf in (log_clf, rnd_clf, bag_clf):
                start_time = time.time()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = Parser.display(y_test, y_pred, clf, start_time)
                accuracies.append(accuracy)

            # Plotting
            classifiers = ['Logistic Regression', 'Random Forest', 'Bagging']
            plt.bar(classifiers, accuracies)
            plt.ylabel('Accuracy (%)')
            plt.title('Classifier Accuracy Comparison')
            plt.show()

        except Exception as exception:
            print(f"Failed to run the implementation due to: {exception}\n")

if __name__ == "__main__":
    Parser.parse_method()