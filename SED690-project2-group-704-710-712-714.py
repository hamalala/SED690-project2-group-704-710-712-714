import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def TrainAlg1(X_train, y_train, X_test, y_test):
  from sklearn.model_selection import GridSearchCV
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import confusion_matrix, classification_report

  # Define the pipeline
  model = Pipeline(steps=[
      ('scaler', StandardScaler()),  # Normalization step
      ('rf', RandomForestClassifier(random_state=42))  # You can replace this with any other classifier
  ])

  param_grid = {
      'rf__n_estimators': [10, 100, 1000],
      'rf__max_depth': [None, 10, 20],
  }


  # Create GridSearchCV
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # You can adjust cv (cross-validation) as needed

  # Fit the pipeline with GridSearchCV
  grid_search.fit(X_train, y_train)

  # Access the best parameters and best estimator
  best_params = grid_search.best_params_
  best_estimator = grid_search.best_estimator_
  print(best_params)
  print(best_estimator)


  # Define the pipeline
  model = Pipeline(steps=[
      ('scaler', StandardScaler()),  # Normalization step
      ('rf', RandomForestClassifier(random_state=42, max_depth=best_params['rf__max_depth'], n_estimators=best_params['rf__n_estimators']))  # You can replace this with any other classifier
  ])

  # Fit the pipeline
  model.fit(X_train, y_train)




  # Make predictions
  y_pred = model.predict(X_test)

  summary_eval = classification_report(y_test,y_pred,digits=4)
  print(summary_eval)

  # Calculate the confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  # Plot the confusion matrix using seaborn heatmap
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', )
  plt.title('Confusion Matrix: Random Forest')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  # Save the plot as an image file (e.g., PNG)
  plt.savefig('confusion_matrix_randforest.png')

  plt.show()

def TrainAlg2(X_train, y_train, X_test, y_test):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestClassifier


  # Define the pipeline with Naive Bayes
  model_nb = Pipeline(steps=[
      ('scaler', StandardScaler()),  # Normalization step
      ('naive_bayes', GaussianNB())  # Naive Bayes model
  ])

  # Set up the parameter grid for Grid Search (Naive Bayes doesn't have many parameters)
  param_grid = {
      # GaussianNB ไม่มีพารามิเตอร์ที่สำคัญมากนัก
  }

  # Perform Grid Search with Cross-Validation (ในกรณีนี้จะใช้ model เรียบง่าย)
  grid_search = GridSearchCV(model_nb, param_grid, cv=5, scoring='accuracy')
  grid_search.fit(X_train, y_train)

  # Get the Best Parameters and Model
  best_params = grid_search.best_params_
  best_estimator = grid_search.best_estimator_

  print(best_params)
  print(best_estimator)

  # Define the pipeline
  model_nb = Pipeline(steps=[
      ('scaler', StandardScaler()),  # Normalization step
      ('naive_bayes', GaussianNB())  # You can replace this with any other classifier
  ])

  # Fit the pipeline
  model_nb.fit(X_train, y_train)

  y_pred = model_nb.predict(X_test)

  summary_eval = classification_report(y_test,y_pred,digits=4)
  print(summary_eval)

  # Calculate the confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  # Plot the confusion matrix using seaborn heatmap
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', )
  plt.title('Confusion Matrix: Naive bayes')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  # Save the plot as an image file (e.g., PNG)
  plt.savefig('confusion_matrix_NaiveBayes.png')

  plt.show()

def TrainAlg3(X_train, y_train, X_test, y_test):
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler

  # Output model performance
  from sklearn.metrics import classification_report, confusion_matrix

  # Define the pipeline with the best parameters
  model_lr = Pipeline(steps=[
      ('scaler', StandardScaler()),  # Normalization step
      ('log_reg', LogisticRegression(
          C=0.1,                # Regularization strength
          penalty='l2',         # L2 norm used in penalization
          solver='liblinear',   # Solver for optimization
          random_state=42,
          max_iter=10000        # Allow more iterations if needed
      ))
  ])

  # Fit the pipeline with the training data
  model_lr.fit(X_train, y_train)

  # Evaluate the model on the test set
  y_pred = model_lr.predict(X_test)

  # Classification report
  print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

  # Confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.title('Confusion Matrix: Logistic Regression')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()


# Set title
st.title("Imbalance")
st.write("Group: 704-710-712-714")
st.write("\n\n")
st.write("Enter a google sheet url")
url = st.text_input("Enter a google sheet url", value="https://drive.google.com/file/d/1HGC-GPDppxEBw7V_2ToxmDxWg1kUebZG")
st.write("Minimum Correlation")
number = st.number_input("Minimum Correlation", min_value=0.0, max_value=1.0, step=0.05)


if st.button("Run Algorithm"):
    df = pd.read_csv(url + "export?format=csv", header=None)
    df.head()
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"'{column}': {unique_values}")
        
