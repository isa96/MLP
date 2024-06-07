import os
import yaml 
import neptune 
import joblib
from dotenv import load_dotenv

from lib.loader import load_pickle
from lib.params import convert_params

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


if __name__ == "__main__":
    
    load_dotenv()

    project_name = os.getenv('NEPTUNE_PROJECT_NAME')
    api_key = os.getenv('NEPTUNE_API_TOKEN')

    run = neptune.init_run(project=project_name)
    
    train_data = load_pickle("train_data.pickle")
    test_data = load_pickle("test_data.pickle")

    train_x = train_data['x']
    train_y = train_data['y']

    test_x = test_data['x']
    test_y = test_data['y']

    with open("config/param.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_version = config["model_versions"]  

    model_mapping = {
        "Logistic Regression": LogisticRegression,
        "SVM": SVC,
    }

    print(train_x.shape, train_y.shape)

    for method in config["methods"]:
        model_name = method["name"]
        model_config = convert_params(method["config"])
        
        model_namespace = f"models/{model_name}/{model_version}"
        
        for param_name, param_value in model_config.items():
            run[f"{model_namespace}/parameters/{param_name}"] = param_value
        
        ModelClass = model_mapping[model_name]
        model = ModelClass(**model_config)
        model.fit(train_x, train_y)

        model_filename = f"{model_name}_{model_version}.joblib"
        joblib.dump(model, model_filename)

        run[f"{model_namespace}/artifact"].upload(model_filename)
        
        predicted_y = model.predict(test_x)
        report = classification_report(test_y, predicted_y)
        
    
        run[f"{model_namespace}/classification_report"] = report
        
        print(f"Classification Report for {model_name} (Version {model_version}):\n{report}\n")
    
    run.stop()

