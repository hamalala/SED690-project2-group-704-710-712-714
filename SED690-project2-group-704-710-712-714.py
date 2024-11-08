import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import pickle

# กำหนด URL หรือเส้นทางของภาพพื้นหลัง 
background_image_url = "https://wallpaperboat.com/wp-content/uploads/2020/12/14/63760/gears-28-920x518.jpg"
# กำหนดสีที่ต้องการ 
text_color = "#000000" 
# สีที่คุณต้องการ 
# ใส่ CSS สำหรับพื้นหลังและสีตัวอักษร 
st.markdown( f""" <style> .stApp {{ background-image: url('{background_image_url}'); background-size: cover; background-position: center; height: 100vh; }} h1, h2, h3, p, div {{ color: {text_color} !important; }}.block-container{{background-color: rgb(225 225 225 / 50%);box-shadow: 2px 5px 15px rgba(0, 0, 0, 0.3);}} </style> """, unsafe_allow_html=True )

def TrainAlg1(name, X_train, y_train, X_test, y_test):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

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

    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model, file)


    summary_eval = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # Convert the report dictionary to a DataFrame
    summary_df = pd.DataFrame(summary_eval).transpose()

    # Display the DataFrame as a table in Streamlit
    st.write("Classification Report:")
    st.table(summary_df)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', )
    plt.title('Confusion Matrix: Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
  
    plt.show()
    st.pyplot(plt)
    rt = dict(name= name, model=model, summary_eval=summary_eval)
    return rt

def TrainAlg2(name, X_train, y_train, X_test, y_test):
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

    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model_nb, file)

    summary_eval = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # Convert the report dictionary to a DataFrame
    summary_df = pd.DataFrame(summary_eval).transpose()

    # Display the DataFrame as a table in Streamlit
    st.write("Classification Report:")
    st.table(summary_df)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', )
    plt.title('Confusion Matrix: Naive bayes')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()
    st.pyplot(plt)

    rt = dict(name= name, model=model_nb, summary_eval=summary_eval)
    return rt

def TrainAlg3(name, X_train, y_train, X_test, y_test):
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

    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model_lr, file)

    summary_eval = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # Convert the report dictionary to a DataFrame
    summary_df = pd.DataFrame(summary_eval).transpose()

    # Display the DataFrame as a table in Streamlit
    st.write("Classification Report:")
    st.table(summary_df)
    

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    st.pyplot(plt)

    rt = dict(name= name, model=summary_df, summary_eval=summary_eval)
    return rt

def PCAPlot (X, y) :
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with PCA components and target variable
    pca_df = pd.DataFrame(data={'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'target': y})

    # Plot the PCA components
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=pca_df, palette='viridis', s=50)
    plt.title('PCA of Features Colored by Target Variable')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Target', loc='upper right')
    plt.show()
    st.pyplot(plt)

# Set title
st.title("Automate Imbalance Training")
st.write("Group: 704-710-712-714")
st.write("\n\n")
# url = st.text_input("Enter a google sheet url", value="https://docs.google.com/spreadsheets/d/1KAGq9A2ppV1aU4WbvDsIq6ATqH7Nehy6PixRpEIO_L8")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
remove_columns_txt = st.text_input("Remove Column before process", value="NO,ID")
threshold = st.number_input("Minimum Correlation", min_value=0.00, max_value=1.000, step=0.05, format="%.3f", value=0.005)
iqr_columns_txt = st.text_input("Request IQR process column", value="INCOME")
target_column = st.text_input("Target column", value="TARGET")


class0_label = st.text_input("Class 0 label", value="Non-fraudulent")
class1_label = st.text_input("Class 1 label", value="Fraudulent")


if 'run_algorithm' not in st.session_state:
    st.session_state.run_algorithm = False


if st.button("Run Algorithm"):
    st.session_state.run_algorithm = True

if st.session_state.run_algorithm:
    if uploaded_file is not None:
        try:
            # Attempt to read the CSV file
            # df = pd.read_csv(url + "/export?format=csv")
            df = pd.read_csv(uploaded_file)
            label_encoders = {}

            st.write("Data Types of Each Column:")
            data_types = df.dtypes  # Get the data types of the columns

            # Alternatively, you can display as a DataFrame for better formatting
            data_types_df = pd.DataFrame(data_types).reset_index()
            data_types_df.columns = ['Column', 'Data Type']
            st.write(data_types_df)

            # Initialize the column configuration dictionary
            column_config = {}

            # Iterate over each column in the DataFrame
            for column in df.columns:
                unique_values_list = [] 
                unique_values = df[column].unique()
                if len(unique_values) < 10:
                    column_config[column] = {
                        'Data Type': df[column].dtype,  # Optionally add the data type from the DataFrame
                        'Unique Values': unique_values.tolist()
                    }
                else :
                    column_config[column] = {
                        'Data Type': df[column].dtype,  # Optionally add the data type from the DataFrame
                        'Unique Values': []
                    }


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
            st.write("---")



            st.write("**** Null value removing ****")
            null_summary = df.isnull().sum()  # Count nulls in each column
            null_columns = null_summary[null_summary > 0]  # Filter columns with nulls

            if not null_columns.empty:
                for column, count in null_columns.items():
                    st.write(f"- {column}: {count} null values")            
                # Drop rows with any null values
                df = df.dropna()  # Drop rows with any null values
            st.write("**** Null value removed ****")
            st.write("---")


            
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
                    df = df[(df[iqr_column] < lower_bound) | (df[iqr_column] > upper_bound)]

            st.write("**** IQR Processed ****")
            st.write("---")



            
            st.write("**** Label Encoder Processing ****")

            text_columns = df.select_dtypes(include=['object']).columns.tolist()  # Get text columns

            if text_columns:            
                # Encode each text column
                for column in text_columns:
                    st.write("Label Encoding Column: " + column)  # Join list elements with a comma            
                    le = LabelEncoder().fit(df[column])
                    df[column] = le.transform(df[column])  # Encode the text column
                    label_encoders[column] = le  # Save the encoder for later use

            st.write("**** Label Encoder Processed ****")
            st.write("---")



            
            st.write("**** Correlation Processing ****")
            # Calculate the correlation matrix
            correlation_matrix = df.corr()

            # Set up the matplotlib figure
            plt.figure(figsize=(10, 8))

            # Create a heatmap using seaborn to visualize the correlation matrix
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, annot_kws={"size": 7})

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

            # Format the correlation values to two decimal places
            correlation_df['Correlation'] = correlation_df['Correlation'].map(lambda x: f"{x:.3f}")

            # Drop duplicate entries if necessary
            correlation_df = correlation_df.drop_duplicates(subset=["Feature"]).reset_index(drop=True)

            # Display the features and their correlation values in a table
            if not correlation_df.empty:
                st.write(f"Features with correlation greater than {threshold}:")
                st.table(correlation_df)
            else:
                st.write(f"No features found with correlation greater than {threshold}.")


            st.write("**** Correlation Processed ****")

            features = [item["Feature"] for item in correlation_data]
            X = df[features]
            y = df[target_column]


            st.write("Cleaned Data")
            st.write("---")
            st.write("---")
            


            model_list = []

            st.markdown("## With out imbalance")
            value_counts = pd.Series(y).value_counts()
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = ['Value', 'Count']
            st.write("Value Counts of Target Column:")
            st.table(value_counts_df)

            PCAPlot(X, y)


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.markdown("### Alg1: RandomForest")
            model_list.append(TrainAlg1('RandomForest with out imbalance', X_train, y_train, X_test, y_test))

            st.markdown("### Alg2: Naive bayes")
            model_list.append(TrainAlg2('Naive bayes with out imbalance',X_train, y_train, X_test, y_test))

            st.markdown("### Alg3: Logistic Regression ")
            model_list.append(TrainAlg3('Logistic Regression with out imbalance',X_train, y_train, X_test, y_test))



            st.write("---")
            # OverSampling
            st.markdown("## With imbalance : OverSampling")
           
            # สร้างตัวอย่างเพิ่มโดยการทำ Oversampling
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            value_counts = pd.Series(y_res).value_counts()
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = ['Value', 'Count']
            st.write("Value Counts of Target Column:")
            st.table(value_counts_df)

            PCAPlot(X_res, y_res)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            st.markdown("### Alg1: RandomForest")
            model_list.append(TrainAlg1('RandomForest with OverSampling',X_train, y_train, X_test, y_test))

            st.markdown("### Alg2: Naive bayes")
            model_list.append(TrainAlg2('Naive bayes with OverSampling', X_train, y_train, X_test, y_test))

            st.markdown("### Alg3: Logistic Regression ")
            model_list.append(TrainAlg3('Logistic Regression with OverSampling', X_train, y_train, X_test, y_test))



            st.write("---")
            #SMOTE
            st.markdown("## With imbalance : SMOTE")
           
            # สร้างตัวแปร SMOTEENN
            smote_enn = SMOTEENN(random_state=42)
            X_res, y_res = smote_enn.fit_resample(X, y)
            value_counts = pd.Series(y_res).value_counts()
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = ['Value', 'Count']
            st.write("Value Counts of Target Column:")
            st.table(value_counts_df)

            PCAPlot(X_res, y_res)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            st.markdown("### Alg1: RandomForest")
            model_list.append(TrainAlg1('RandomForest with SMOTE', X_train, y_train, X_test, y_test))

            st.markdown("### Alg2: Naive bayes")
            model_list.append(TrainAlg2('Naive bayes with SMOTE',X_train, y_train, X_test, y_test))

            st.markdown("### Alg3: Logistic Regression ")
            model_list.append(TrainAlg3('Logistic Regression with SMOTE',X_train, y_train, X_test, y_test))



            st.write("---")
            #Undersampling
            st.markdown("## With imbalance : Undersampling")
           
            # สร้างตัวแปร Undersampling
            rus = RandomUnderSampler()
            # Apply Tomek links undersampling
            X_res, y_res = rus.fit_resample(X, y)
            value_counts = pd.Series(y_res).value_counts()
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = ['Value', 'Count']
            st.write("Value Counts of Target Column:")
            st.table(value_counts_df)

            PCAPlot(X_res, y_res)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            st.markdown("### Alg1: RandomForest")
            model_list.append(TrainAlg1('RandomForest with Undersampling', X_train, y_train, X_test, y_test))

            st.markdown("### Alg2: Naive bayes")
            model_list.append(TrainAlg2('Naive bayes with Undersampling', X_train, y_train, X_test, y_test))

            st.markdown("### Alg3: Logistic Regression ")
            model_list.append(TrainAlg3('Logistic Regression with Undersampling', X_train, y_train, X_test, y_test))


            # Assuming model_list is already filled with the models and their evaluation summaries
            results = []

            for model_info in model_list:
                name = model_info['name']
                summary_eval = model_info['summary_eval']
                
                # Extract metrics from the macro average
                metrics = {
                    'Model': name,
                    'Accuracy': summary_eval['accuracy'],
                    'Precision': summary_eval['macro avg']['precision'],
                    'Recall': summary_eval['macro avg']['recall'],
                    'F1-Score': summary_eval['macro avg']['f1-score']
                }
                
                results.append(metrics)

            # Create a DataFrame from the results
            results_df = pd.DataFrame(results)

            

            # Create a heatmap for the metrics
            plt.figure(figsize=(10, 6))
            sns.heatmap(results_df.set_index('Model').T, annot=True, fmt=".4f", cmap='coolwarm', cbar=True)
            plt.title('Model Performance Metrics')
            plt.xlabel('Models')
            plt.ylabel('Metrics')
            plt.show()
            st.pyplot(plt)

            results = []

            for model_info in model_list:
                name = model_info['name']
                model = model_info['model']
                summary_eval = model_info['summary_eval']
                
                # Extract metrics from the macro average
                metrics = {
                    'Model name': name,
                    'Accuracy': summary_eval['accuracy'],
                    'Precision': summary_eval['macro avg']['precision'],
                    'Recall': summary_eval['macro avg']['recall'],
                    'F1-Score': summary_eval['macro avg']['f1-score'],
                    'Model': model
                }
                
                results.append(metrics)

            # Create a DataFrame from the results
            results_df = pd.DataFrame(results)

            st.write("---")
            st.write("---")
            # Display the DataFrame
            st.write("### Model Performance Results")
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])
            with col1:
                st.write("**Model Name**")
            with col2:
                st.write("**Accuracy**")
            with col3:
                st.write("**Precision**")
            with col4:
                st.write("**Recall**")
            with col5:
                st.write("**F1-Score**")
            with col6:
                st.write("**Download**")

            for index, row in results_df.iterrows():

                data_to_save = {
                    'Model name': row['Model name'],
                    'Accuracy': row['Accuracy'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall'],
                    'F1-Score': row['F1-Score'],

                    'model': row['Model'],
                    'data_types': column_config,
                    "features": features,
                    "target": target_column,
                    
                    "label_encoders": label_encoders,
                    "class0_label": class0_label,
                    "class1_label": class1_label
                }

                # Save the model and JSON data into a .pkl file
                with open(f'{row['Model name']}.pkl', 'wb') as file:
                    pickle.dump(data_to_save, file)


                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])
                with col1:
                    st.write(row['Model name'])
                with col2:
                    st.write(f"{row['Accuracy']:.3f}")
                with col3:
                    st.write(f"{row['Precision']:.3f}")
                with col4:
                    st.write(f"{row['Recall']:.3f}")
                with col5:
                    st.write(f"{row['F1-Score']:.3f}")
                with col6:
                    with open(f'{row['Model name']}.pkl', 'rb') as file:
                        btn = st.download_button(
                            label="Download Model and config",
                            data=file,
                            file_name=f"{row['Model name']}.pkl",
                            mime="application/octet-stream"
                        )


                        

        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
    else:
        st.warning("Please upload a CSV file before clicking the button.")
