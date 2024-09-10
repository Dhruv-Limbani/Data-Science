import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. Set up the Title
st.title("Credit Risk Analysis and Modeling")

# 2. Upload Dataset
st.header("Step 1: Load Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Show the first few rows

    # 3. Check Missing Values
    st.header("Step 2: Check for Missing Values")
    st.write(df.isnull().sum())

    # 4. Exploratory Data Analysis (EDA)
    st.header("Step 3: Exploratory Data Analysis")
    
    # Visualize numerical attributes with histograms
    st.subheader("Numerical Attributes")
    if st.checkbox("Show histograms"):
        numerical_cols = ['duration', 'credit_amount', 'age']  # Change this to actual columns
        for col in numerical_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # Visualize categorical attributes with bar plots
    st.subheader("Categorical Attributes")
    if st.checkbox("Show bar plots"):
        categorical_cols = ['checking_status', 'credit_history', 'purpose']  # Change this to actual columns
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            st.pyplot(fig)

    # 5. Handling Data Imbalance using SMOTE
    st.header("Step 4: Handle Data Imbalance")
    if st.button("Apply SMOTE"):
        X = df.drop('class', axis=1)  # Replace 'class' with your target variable
        y = df['class']  # Replace with your actual target
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X, y)
        st.write("Data after applying SMOTE:")
        st.write(pd.Series(y_res).value_counts())

    # 6. Train-Test Split
    st.header("Step 5: Train-Test Split")
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=42)
    st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 7. Model Building and Hyperparameter Tuning
    st.header("Step 6: Model Building and Tuning")
    if st.checkbox("Run Hyperparameter Tuning"):
        # Example with RandomForest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
        clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        st.write("Best Parameters found:")
        st.write(grid_search.best_params_)
        
        y_pred = grid_search.predict(X_test)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    # 8. Conclusion
    st.header("Step 7: Conclusion")
    st.write("The best model and its performance metrics are displayed above.")
