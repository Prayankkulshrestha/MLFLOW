#src/main.py
from numpy.lib import utils
from utils import data_utils,data_prep,model,config
import os,sys





if __name__ == "__main__":

    model_name = sys.argv[1]
    print(f"Model name is {model_name}")


    file_path = os.path.join(os.getcwd(),"input","Telco-Customer-Churn.csv")
    data = data_utils.utils.load_data(file_path)
    
    # check the data Quality
    data_utils.utils.data_check(data)

    # data Preparation
    prep_obj = data_prep.data_preparation(data)
    clean_df = prep_obj.data_prep_()

    model_obj = model.Model(model_name)
    (experimentID, runID) = model_obj.mlflow_run(clean_df)
    print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
    print("-" * 100)

