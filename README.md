# income-prediction-machine-learning-project
This code appears to be performing various tasks related to data preprocessing, exploration, feature selection, and machine learning using the Pandas library for data manipulation and scikit-learn for machine learning. I'll break down each part of the code for you:

1. **Data Loading**:
   ```python
   import pandas as pd
   df = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")
   ```
   This code imports the Pandas library and loads a CSV file into a DataFrame called `df`. The CSV file path is specified as "/kaggle/input/adult-income-dataset/adult.csv."

2. **Data Exploration**:
   ```python
   df.education.value_counts()
   df.occupation.value_counts()
   ```
   These lines display the value counts for the 'education' and 'occupation' columns in the DataFrame, which can help in understanding the distribution of these categorical variables.

3. **One-Hot Encoding**:
   ```python
   # One-hot encoding categorical columns
   df = pd.concat([df.drop("occupation", axis=1), pd.get_dummies(df['occupation'], dtype=int).add_prefix("occupation_")], axis=1)
   df = pd.concat([df.drop("workclass", axis=1), pd.get_dummies(df['workclass'], dtype=int).add_prefix("workclass_")], axis=1)
   df = df.drop("education", axis=1)
   df = pd.concat([df.drop("marital-status", axis=1), pd.get_dummies(df['marital-status'], dtype=int).add_prefix("marital-status_")], axis=1)
   df = pd.concat([df.drop("relationship", axis=1), pd.get_dummies(df['relationship'], dtype=int).add_prefix("relationship_")], axis=1)
   df = pd.concat([df.drop("race", axis=1), pd.get_dummies(df['race'], dtype=int).add_prefix("race_")], axis=1)
   df = pd.concat([df.drop("native-country", axis=1), pd.get_dummies(df['native-country'], dtype=int).add_prefix("native-country_")], axis=1)
   ```
   This code performs one-hot encoding for various categorical columns in the DataFrame. It creates binary (0 or 1) columns for each category within these categorical columns, effectively converting them into a numerical format for machine learning.

4. **Binary Encoding**:
   ```python
   df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
   df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
   ```
   This code converts binary categorical variables ('gender' and 'income') into numerical format (1 for 'Male' and '>50K', and 0 otherwise).

5. **Correlation Analysis**:
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   plt.figure(figsize=(18, 12))
   sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
   ```
   These lines create a heatmap to visualize the correlation between different features in the DataFrame. This can help identify which features are strongly correlated with the target variable 'income'.

6. **Feature Selection**:
   ```python
   correlations = df.corr()['income'].abs()
   sorted_correlations = correlations.sort_values()
   num_cols_to_drop = int(0.8 * len(df.columns))
   cols_to drop = sorted_correlations.iloc[:num_cols_to_drop].index
   df_dropped = df.drop(cols_to_drop, axis=1)
   ```
   This code calculates the absolute correlations between features and the target variable ('income'). It then drops a certain percentage of features (80% in this case) with the lowest absolute correlations.

7. **Machine Learning Model Preparation**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   df = df.drop('fnlwgt', axis=1)
   train_df, test_df = train_test_split(df, test_size=0.2)
   ```
   These lines import the RandomForestClassifier from scikit-learn, drop the 'fnlwgt' column from the DataFrame, and split the data into training and testing sets.

8. **Model Training**:
   ```python
   train_x = train_df.drop("income", axis=1)
   train_y = train_df['income']
   test_x = test_df.drop("income", axis=1)
   test_y = test_df['income']
   forest = RandomForestClassifier()
   forest.fit(train_x, train_y)
   ```
   Here, a Random Forest Classifier is trained on the training data.

9. **Model Evaluation**:
   ```python
   forest.score(test_x, test_y)
   ```
   This line calculates the accuracy score of the trained model on the test data.

10. **Feature Importance**:
    ```python
    importance = dict(zip(forest.feature_names_in_, forest.feature_importances_))
    importance = {k: v for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}
    ```
    These lines calculate and rank the feature importances based on the trained Random Forest model.

11. **Hyperparameter Tuning with Grid Search**:
    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators": [50, 100, 250],
        "max_depth": [5, 20, 30, None],
        "min_samples_split": [2, 4],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose=10)
    grid_search.fit(train_x, train_y)
    ```
    This code performs hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters for the Random Forest model.

12. **Best Model Selection and Evaluation**:
    ```python
    forest = grid_search.best_estimator_
    forest.score(test_x, test_y)
    ```
    Here, the best model from the grid search is selected, and its accuracy is evaluated on the test data.

13. **Final Feature Importance**:
    ```python
    importance = dict(zip(forest.feature_names_in_, forest.feature_importances_))
    importance = {k: v for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}
    ```
    These lines calculate and rank the feature importances for the best-selected model.

The code is essentially a data preprocessing and machine learning pipeline for a classification task, with a focus on feature engineering and model selection. It includes data exploration, encoding, feature selection, model training, evaluation, and hyperparameter tuning.
