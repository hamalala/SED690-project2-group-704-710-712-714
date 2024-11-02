import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


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
url = st.text_input("Enter a google sheet url", value="https://docs.google.com/spreadsheets/d/1KAGq9A2ppV1aU4WbvDsIq6ATqH7Nehy6PixRpEIO_L8")
remove_columns_txt = st.text_input("Remove Column before process", value="NO,ID")
threshold = st.number_input("Minimum Correlation", min_value=0.00, max_value=1.000, step=0.05, value=0.01)
iqr_columns_txt = st.text_input("Request IQR process column", value="INCOME")
target_column = st.text_input("Target column", value="TARGET")

if st.button("Run Algorithm"):
    st.write("Analize Data from " + url + "/export?format=csv")
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(url + "/export?format=csv")
        label_encoders = {}

        st.write("Data Types of Each Column:")
        data_types = df.dtypes  # Get the data types of the columns

        # Alternatively, you can display as a DataFrame for better formatting
        data_types_df = pd.DataFrame(data_types).reset_index()
        data_types_df.columns = ['Column', 'Data Type']
        st.write(data_types_df)

        # Display a sample of the data to confirm it's loaded correctly
        # st.write("Sample Data")
        # st.write(df.head())

        st.write("Start data cleansing")


        st.write("**** Unused columns removing ****")
        if remove_columns_txt != '' :
            remove_columns = remove_columns_txt.split(",")            
            # Iterate over each fruit in the list
            for remove_column in remove_columns:
                df = df.drop(remove_column,axis=1)
                st.write("Column '" + remove_column + "' removed")
        st.write("**** Unused columns removed ****")


        st.write("**** Null value removing ****")
        null_summary = df.isnull().sum()  # Count nulls in each column
        null_columns = null_summary[null_summary > 0]  # Filter columns with nulls

        if not null_columns.empty:
            for column, count in null_columns.items():
                st.write(f"- {column}: {count} null values")            
            # Drop rows with any null values
            df = df.dropna()  # Drop rows with any null values
        st.write("**** Null value removed ****")


        
        st.write("**** IQR Processing ****")
        if iqr_columns_txt != '':
            iqr_columns = iqr_columns_txt.split(",")            
            # Iterate over each column in the list
            for iqr_column in iqr_columns:
                # Calculate Q1, Q3, and IQR
                Q1 = df[iqr_column].quantile(0.25)
                Q3 = df[iqr_column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Convert numerical values to strings for concatenation
                st.write(f"{iqr_column} Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower bound: {lower_bound}, Upper bound: {upper_bound};")

                # Remove outliers from the DataFrame
                df = df[(df[iqr_column] >= lower_bound) & (df[iqr_column] <= upper_bound)]

        st.write("**** IQR Processed ****")


        
        st.write("**** Label Encoder Processing ****")
        text_columns = df.select_dtypes(include=['object']).columns.tolist()  # Get text columns

        if text_columns:            
            # Encode each text column
            for column in text_columns:
                st.write("Label Encoding Column: " + column)  # Join list elements with a comma            
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])  # Encode the text column
                label_encoders[column] = le  # Save the encoder for later use

        st.write("**** Label Encoder Processed ****")



        
        st.write("**** Correlation Processing ****")
        # Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))

        # Create a heatmap using seaborn to visualize the correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

        # Show the plot
        plt.title('Correlation Matrix of Features')
        plt.show()
        st.pyplot(plt)

       
        # Get the correlation of all features with the target column
        target_correlation = correlation_matrix[target_column]

        # Filter features with absolute correlation greater than or equal to the threshold
        features_above_threshold = target_correlation[abs(target_correlation) >= threshold]

        # Create a list to store feature names and their correlation values
        correlation_data = []

        # Populate the correlation_data list with dictionaries
        for feature, correlation_value in features_above_threshold.items():
            if feature != target_column:  # Exclude the target column itself
                correlation_data.append({"Feature": feature, "Correlation": correlation_value})

        # Convert the list to a DataFrame
        correlation_df = pd.DataFrame(correlation_data)

        # Drop duplicate entries if necessary
        correlation_df = correlation_df.drop_duplicates(subset=["Feature"]).reset_index(drop=True)

        # Display the features and their correlation values in a table
        if not correlation_df.empty:
            st.write(f"Features with correlation greater than {threshold}:")
            st.table(correlation_df)
        else:
            st.write(f"No features found with correlation greater than {threshold}.")


        st.write("**** Correlation Processed ****")


        

       

        st.write("Cleaned Data:")
        st.write(df.head())

    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        
