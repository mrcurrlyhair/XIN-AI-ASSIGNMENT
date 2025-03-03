This model predicits wether 3 or more verchals are involved in a traffic accdient. 3 or more cars are/will be classed as highrisk for injury. 
models used are random forrest, xgboost and gradient boosting. The best model will be chosen based of f1 score and how accurate they are compared to testing data. 

run clean_data.py first , should create new file of cleaned_traffic_accidents.csv
run pre_graphs.py, this will show graphs of data comparable data in barcharts and heatmaps
run models.py, this will train the models and output accuracy and f1 score for all models (!!!!PLEASE NOTE GRADIENT BOOSTING MODEL WILL TAKE TIME TO RUN PLEASE WAIT!!!!)

Alterntivley but do also run models_matrix.py , this will output the accuracy and f1scores but also output confusion matrixes of the models
run models_output.py, this will output graphs using the model 


