from sklearn.ensemble import GradientBoostingRegressor



#########################
# Downstream GBDT model #
#########################

def separate_train_gbdt_predictor(dataset, ahead, n_estimators=100, max_depth=3, max_features=None):
  """ Separately train the GBDT regression for prediction """
  X, y = dataset[:, :-ahead], dataset[:, -ahead:].ravel()
  model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
  model.fit(X, y)
  return model


def prediction_gbdt(model, dataset, ahead, scaler=None):
  """ Prediction error of downstream GBDT model """
  X_test, y_test = dataset[:, :-ahead], dataset[:, -ahead:]
  y_pred = model.predict(X_test)
  if scaler is not None:
    y_test = scaler.inverse_transform(y_test.reshape(1, -1))
    y_pred = scaler.inverse_transform(y_pred.reshape(1, -1))
  return y_test.ravel(), y_pred.ravel()


def count_params_gbdt(model):
  """ Count the number of parameters in the downstream GBDT model """
  return sum(est.tree_.node_count for est in model.estimators_.flatten())









