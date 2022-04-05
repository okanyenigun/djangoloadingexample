from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt

class Regression:
    
    def __init__(self):
        #generate dataset
        X, y = self._generate_dataset()
        #run model
        self.rmse = self._run_model(X,y)
        
    def get_rmse(self):
        """returns root mean square"""
        return self.rmse

    def _generate_dataset(self):
        """generates dummy dataset using sklearn method"""
        X,y = datasets.make_regression(n_samples=1000000,#number of samples
                                      n_features=15,#number of features
                                      n_informative=10,#number of useful features 
                                      noise=10,#bias and standard deviation of the guassian noise
                                      random_state=0) #set for same data points for each run
        return X,y

    def _run_model(self,X,y):
        """lasso model is tuned by gridsearch, returns root mean square of best models"""
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
        model = Lasso()
        param_grid = {
            "max_iter": [100,1000,10000],
            "alpha": [0.00001,0.0001,0.001]
        }
        cv = GridSearchCV(model,param_grid=param_grid,n_jobs=-1,cv=5,scoring='neg_mean_absolute_error')
        cv.fit(X_train,y_train)
        best_model = cv.best_estimator_
        y_pred = best_model.predict(X_test)
        rms = sqrt(mean_squared_error(y_test,y_pred))
        return rms