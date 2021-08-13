from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


MLFLOW_PATH = r'C:\Users\Prayank.Kulshrestha\Documents\MLFLOW\artifact'

 
model_dict = {
                  "KNN":KNeighborsClassifier(),
                  "Decision_tree": tree.DecisionTreeClassifier(),
                  "RandomForest":ensemble.RandomForestClassifier(n_estimators=200),
                  "Adaboost":ensemble.AdaBoostClassifier(learning_rate=0.5),
                  "SVM":SVC()
                  }



model_params = {
                    "KNN":{"n_neighbors":[7,9,11]},
                    "Decision_tree" :{"max_depth":[12,14,16]},
                    "RandomForest" :{"max_depth":[12,14,16]},
                    "Adaboost":{"n_estimators":[200,300,500]},
                    "SVM":{"C":[0.05,1,100]}
                    }