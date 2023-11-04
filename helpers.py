from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Lets create a function that splits our data into a 60/40 train/test split
# We also need to make sure the function can accept a random seed
def get_train_test_split(df_X: pd.DataFrame, df_y: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  A function that splits our data into a 60/40 train/test split and accepts a random seed

  Args:
    df_X (pd.DataFrame): The dataframe containing our feature (X) variables
    df_y (pd.DataFrame): The dataframe containing our label (y) variables
    random_seed (int): The random seed to use for reproducibility and experimentation

  Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing X_train, X_test, y_train, y_test
  """
  # Call the sklearn train_test_split function
  return train_test_split(df_X, df_y, test_size=0.4, random_state=random_seed)


@dataclass
class LinearRegressionMetrics:
  root_mean_squared_error: float
  r_squared: float


@dataclass
class LinearRegressionEvaluation:
  training_metrics: LinearRegressionMetrics
  test_metrics: LinearRegressionMetrics

  def __str__(self):
    return f"Training Metrics: [RMSE: {self.training_metrics.root_mean_squared_error}, R-squared: {self.training_metrics.r_squared}]\nTest Metrics: [RMSE: {self.test_metrics.root_mean_squared_error}, R-squared: {self.test_metrics.r_squared}]"


@dataclass
class LinearRegressionExperiment:
  """
  A class that represents a linear regression experiment.

  This class is able to:
  - Generate a train/test split
  - Fit a linear regression model
  - Evaluate the model
  """
  df_X: pd.DataFrame
  df_y: pd.DataFrame
  experiment_number: int

  model: LinearRegression = field(init=False)
  X_train: pd.DataFrame = field(init=False)
  X_test: pd.DataFrame = field(init=False)
  y_train: pd.DataFrame = field(init=False)
  y_test: pd.DataFrame = field(init=False)

  y_train_pred: np.ndarray = field(init=False)
  y_test_pred: np.ndarray = field(init=False)

  evaluation: LinearRegressionEvaluation = field(init=False)

  def fit_predict_evaluate(self, normalise_X: bool = False):
    # Normalise the data if required
    if normalise_X:
      scaler = StandardScaler()
      self.df_X = pd.DataFrame(scaler.fit_transform(self.df_X), columns=self.df_X.columns)

    # First create a train/test split
    X_train, X_test, y_train, y_test = get_train_test_split(df_X=self.df_X, df_y=self.df_y, random_seed=self.experiment_number)
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    # Now fit the model
    self.model = LinearRegression().fit(X_train, y_train)

    # Make predictions
    self.y_train_pred = self.model.predict(X_train)
    self.y_test_pred = self.model.predict(X_test)

    # Calculate the metrics
    training_metrics = LinearRegressionMetrics(
      root_mean_squared_error=mean_squared_error(y_train, self.y_train_pred),
      r_squared=r2_score(y_train, self.y_train_pred)
    )

    test_metrics = LinearRegressionMetrics(
      root_mean_squared_error=mean_squared_error(y_test, self.y_test_pred),
      r_squared=r2_score(y_test, self.y_test_pred)
    )

    self.evaluation = LinearRegressionEvaluation(
      training_metrics=training_metrics,
      test_metrics=test_metrics
    )

  def get_evaluation_as_pd(self, name: str) -> pd.DataFrame:
    df = pd.DataFrame()

    # Add the training data
    df = pd.concat([df, pd.DataFrame({
      'experiment_number': [self.experiment_number],
      'experiment': [name],
      'split': ['train'],
      'rmse': [self.evaluation.training_metrics.root_mean_squared_error],
      'rsquared': [self.evaluation.training_metrics.r_squared],
    })])

    return pd.concat([df, pd.DataFrame({
      'experiment_number': [self.experiment_number],
      'experiment': [name],
      'split': ['test'],
      'rmse': [self.evaluation.test_metrics.root_mean_squared_error],
      'rsquared': [self.evaluation.test_metrics.r_squared],
    })])
