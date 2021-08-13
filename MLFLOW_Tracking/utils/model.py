import mlflow.sklearn
import mlflow
from sklearn import model_selection
from sklearn import metrics

from utils import config


class Model:

    def __init__(self,model_name):

        """
        Constructor for classification models like DecisionClassfication,RandomForestClassification etc
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self.model_name = model_name
        self.model_ = config.model_dict[self.model_name]
        self._params = config.model_params[self.model_name]
        

    @classmethod
    def new_instance(cls,params):
        return cls(params)

    
    @property
    def model(self):
        """
        Getter for the property the model
        :return: return the model
        """
        
        return self._model
    
    @property
    def parameters(self):
        '''
        Getter for property of the hyperparameter
        :return: return the hyperparameter
        '''

        return self._param
    
    def mlflow_run(self, df, r_name="mlflow_classifiction result",):
        '''
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        '''

        with mlflow.start_run(run_name=r_name) as run:
            # get current run id and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # saving process the data before modelling
            mlflow.log_artifact(config.MLFLOW_PATH)

            # get the data
            X = df.drop("Churn",axis=1).values
            y = df['Churn'].values

            # split the data
            xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
            
            param_name = list(self._params.keys())[0]
            for each_param in list(self._params.values())[0]:

                self.model_.param_name = each_param
                self.model_.fit(xtrain,ytrain)

                y_pred = self.model_.predict(xtest)

                # log the model and param using MLflow Sklearn APIs
        
                mlflow.sklearn.log_model(self.model_,self.model_name)
                mlflow.log_params({param_name:each_param})


                # Compute the evalution metrics
                acc = metrics.accuracy_score(y_true=ytest,y_pred=y_pred)
                precision = metrics.precision_score(ytest,y_pred)
                recall = metrics.recall_score(ytest,y_pred)
                roc = metrics.roc_auc_score(ytest,y_pred)

                # get confusion matrix values
                conf_matrix = metrics.confusion_matrix(ytest,y_pred)
                true_positive = conf_matrix[0][0]
                true_negative = conf_matrix[1][1]
                false_positive = conf_matrix[0][1]
                false_negative = conf_matrix[1][0]

                # get classification matrics as a dictionary
                class_report = metrics.classification_report(ytest,y_pred, output_dict=True)
                recall_0 = class_report['0']['recall']
                f1_score_0 = class_report['0']['f1-score']
                recall_1 = class_report['1']['recall']
                f1_score_1 = class_report['1']['f1-score']

                
                # log metrics
                mlflow.log_metric("accuracy_score", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("true_positive", true_positive)
                mlflow.log_metric("true_negative", true_negative)
                mlflow.log_metric("false_positive", false_positive)
                mlflow.log_metric("false_negative", false_negative)
                mlflow.log_metric("recall_0", recall_0)
                mlflow.log_metric("f1_score_0", f1_score_0)
                mlflow.log_metric("recall_1", recall_1)
                mlflow.log_metric("f1_score_1", f1_score_1)
                mlflow.log_metric("roc", roc)


                # print some data
                print("-" * 100)
                print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
                print(conf_matrix)
                print(metrics.classification_report(ytest,y_pred))
                print("Accuracy Score:", acc)
                print("Precision     :", precision)
                print("ROC           :", roc)
                return (experimentID, runID)




    

    

