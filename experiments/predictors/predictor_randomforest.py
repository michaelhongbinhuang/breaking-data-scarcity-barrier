from sklearn.ensemble import RandomForestRegressor



##################################
# Downstream Random Forest Model #
##################################

def separate_train_rf_predictor(dataset, ahead, n_estimators=100, max_depth=5, max_features=None):
  """ Separately train the random forest regression for prediction """
  X, y = dataset[:, :-ahead], dataset[:, -ahead:].ravel()
  model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
  model.fit(X, y)
  return model


def prediction_rf(model, dataset, ahead, scaler=None):
  """ Prediction error of downstream random forest model """
  X_test, y_test = dataset[:, :-ahead], dataset[:, -ahead:]
  y_pred = model.predict(X_test)
  if scaler is not None:
    y_test = scaler.inverse_transform(y_test.reshape(1, -1))
    y_pred = scaler.inverse_transform(y_pred.reshape(1, -1))
  return y_test.ravel(), y_pred.ravel()


def count_params_rf(model):
  """ Count the number of parameters in the downstream random forest model """
  return sum(est.tree_.node_count for est in model.estimators_)









