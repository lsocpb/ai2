from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, criterion, dataset_name, feature_names=None, class_names=None):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    
    # Wizualizacja drzewa
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(dataset_name + "_" + criterion + "_tree", format='png', cleanup=True)
    
    # Wysokość drzewa
    print("Wysokość drzewa z wskaźnikiem", criterion, "dla", dataset_name, ":", clf.tree_.max_depth)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Dokładność klasyfikacji:", accuracy)

def process_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.50, random_state=42), iris.feature_names, iris.target_names

def process_breast_cancer_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.5, random_state=42), None, None

def main():
    # Iris dataset
    (X_train_iris, X_test_iris, y_train_iris, y_test_iris), feature_names_iris, class_names_iris = process_iris_data()
    train_and_evaluate_classifier(X_train_iris, y_train_iris, X_test_iris, y_test_iris, "gini", "iris", feature_names_iris, class_names_iris)
    train_and_evaluate_classifier(X_train_iris, y_train_iris, X_test_iris, y_test_iris, "entropy", "iris", feature_names_iris, class_names_iris)

    # Breast cancer dataset
    (X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer), _, _ = process_breast_cancer_data()
    train_and_evaluate_classifier(X_train_cancer, y_train_cancer, X_test_cancer, y_test_cancer, "gini", "cancer")
    train_and_evaluate_classifier(X_train_cancer, y_train_cancer, X_test_cancer, y_test_cancer, "entropy", "cancer")

if __name__ == "__main__":
    main()
