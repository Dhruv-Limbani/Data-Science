import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from scipy.stats import pointbiserialr

def show_summary(df):
    # Extract summary information manually
    summary = {
        "Column": df.columns,
        "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
        "Dtype": [df[col].dtype for col in df.columns]
    }
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary)

    # Count of columns by dtype
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Dtype", "Column Count"]

    # Display the summary in a table format
    st.header("Summary of the Data:",divider=True)

    col1, col2 = st.columns([3,2])
    with col1:
        st.dataframe(summary_df)

    # Display the count of columns by dtype
    with col2:
        st.write("Count of Columns by Data type:")
        st.dataframe(dtype_counts)

        st.write("Dataset Size: ")
        size_df = {
            "Axis" : ["Samples","Features"],
            "Count": [df.shape[0], df.shape[1]]
        }
        size_df = pd.DataFrame(size_df)
        st.dataframe(size_df)

def show_unique_values(df,columns):
    # Create a list to store the summary data
    uniq_val_data = []
    
    for col in columns:
        dtype = df[col].dtype
        unique_values = df[col].unique()
        strg = ""
        for uv in unique_values[:-1]:
            strg = strg + f"{uv}, "
        strg = strg + f"{unique_values[-1]}"
        
        # Add the column data to the summary list
        uniq_val_data.append({
            "Column": col,
            "Data Type": dtype,
            "Unique Values (sorted)": strg
        })
    
    # Create a DataFrame from the summary data
    uniq_val_df = pd.DataFrame(uniq_val_data)
    
    # Display the summary in a table format
    st.dataframe(uniq_val_df)

    st.success("Findings:")
    st.write("1) The Elements in each column are consistent with their respective column's datatype")
    st.write("2) The column 'credit_history' contains some values eqaul to 'no credits/all paid' and some values equal to 'all paid'. So Values equal to 'no credits/all paid' needs to be replaced to 'no credits' in 'credit_history' column to avoid confusion and maintain uniqueness of each sample.")
    st.write("3) The column 'personal status' can be divided into two columns: one for gender and other for marital status.")


def show_outlier_detection(df, numerical_columns, method):
    
    if method==1:
        for column in numerical_columns:

            fig, axes = plt.subplots(1,2,figsize=(15,4))

            # Box Plot
            sns.boxplot(y=df[column],ax=axes[0])
            axes[0].set_title(f'Box plot of {column}')

            # Histogram
            sns.histplot(df[column], kde=True, bins=30, ax=axes[1])
            axes[1].set_title(f'Histogram of {column}')

            st.pyplot(fig)
    
    if method==0:
        stats = df.groupby("class").describe().T
        st.dataframe(stats)

    st.success("Observations:")
    st.write("1) For duration: 75% of the population with bad credit is having duration of 36 months or less whereas same percent of those with good credit are having duration of 24 months or less.")
    st.write("2) For credit_amount: 75% of the population with bad credit is having credit_amount of 5141.5 or less whereas same percent of those with good credit are having credit_amount of 3634.75 or less.")
    st.write("3) Hence we cannot eleminate samples as these two columns are critical to the target class.")


def bivariate_categorical(df, col1, col2):
    fig, axes = plt.subplots(1,2,figsize=(15,4))
    ct = pd.crosstab(df[col2],df[col1], normalize = 'index')
    # Bar Plot
    st.write("Contingency Table")
    st.dataframe(ct)
    ct.plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'))
    axes[0].set_title(f'Proportion of {col1.title()} by {col2.title()}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel('Proportion')
    axes[0].legend(title=f'{col1.title()}')
    # Stacked Bar Plot
    ct.plot(kind='bar', stacked=True, ax=axes[1], color=sns.color_palette('Set2'))
    axes[1].set_title(f'Stacked Bar Chart of {col1.title()} by {col2.title()}')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title=f'{col1.title()}')
    st.pyplot(fig)

def show_EDA(df, columns, method):
    if method==0:
        column = st.selectbox("Choose variable for Univariate Analysis:", options = categorical_columns)
        if column:
            fig, axes = plt.subplots(1,2,figsize=(15,4))
            # Box Plot
            sns.countplot(x=column, data=df, ax=axes[0])
            axes[0].set_title(f'Box plot of {column}')

            # Histogram
            df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
            axes[1].set_title(f'Histogram of {column}')

            st.pyplot(fig)

            st.dataframe(df[column].value_counts())

    if method==1:
        column = st.selectbox("Choose variable against target class for Bivariate Analysis:", options = columns)
        if column:
            if column in categorical_columns:
                bivariate_categorical(df, column, 'class')
                flip_axis = st.checkbox("Flip axis")
                if flip_axis:
                    bivariate_categorical(df,'class',column)
                
            else:
                fig, axes = plt.subplots(1,2,figsize=(15,4))
                ct = pd.crosstab(df['class'],df[column], normalize = 'index')
                # Histogram
                sns.histplot(data=df, x=column, hue='class', multiple='stack', kde=True, bins=20, ax=axes[0])
                axes[0].set_title(f'Histogram of {column} by Credit')

                # Boxplot
                sns.boxplot(data=df, x='class', y=column, ax=axes[1])
                axes[1].set_title(f'Boxplot of {column} by Credit')

                st.pyplot(fig)
        st.success("Final Findings of Bivariate Analysis both (Numerical and Categorical Variables Combined):")
        st.write("1) Attributes that provided some insights: duration, credit_amount ,age, checking_status, credit_history, purpose, employment, property_magnitude, housing, gender")
        st.write("2) Rest of the attributes may not efficiently distinguish among people with good credit and bad credit.")

    if method==2:
        correlation = df[numerical_columns].corr()
        st.dataframe(correlation)
        fig = plt.figure(figsize=(8,4))
        a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
        a.set_xticklabels(a.get_xticklabels(), rotation=90)
        a = a.set_title('Correlation within Attributes')
        st.pyplot(fig)

        st.success("Conclusion:")
        st.write("1) There is a high positive correlation between duration and credit_amount, so we can use either one of the attribute.")
        st.write("2) There is slightly negative correlation between installment_commitment and credit_amount.")
        st.write("3) A slight positive correlation is also observed between age and residence_since.")


def show_feat_impt(df, columns, method):
    if method==0:
        m = st.selectbox("Choose the ensemble model:",options=['Random Forest Classifier', 'Gradient Boosting Classifier'])
        if m == 'Random Forest Classifier':
            model = RandomForestClassifier(random_state=42)

            model.fit(df[columns], df['class'])

            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': columns, 'Importance': importances})

            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        else:
            model = GradientBoostingClassifier(random_state=42)

            model.fit(df[columns], df['class'])

            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': columns, 'Importance': importances})

            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        fig, axes = plt.subplots(1,2,figsize=(10, 6))
        axes[1].pie(feature_importance_df['Importance'], labels=feature_importance_df['Feature'], autopct='%1.1f%%')
        axes[1].set_title("Feature Importance")
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[0])
        axes[0].set_title("Feature Importance")
        st.pyplot(fig)
    
    if method==1:
        X_encoded = df[columns].drop('class',axis=1).apply(LabelEncoder().fit_transform)
        y_encoded = LabelEncoder().fit_transform(df['class'])
        m = st.selectbox("Choose the method:",options=['Chi-Square Test', 'ANOVA F-value', 'Point Biserial Correlation', 'Mutual Information', 'RandomForestClassifier'])
        if m == 'Chi-Square Test':
            chi2_values, p_values = chi2(X_encoded, y_encoded)
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Chi2': chi2_values, 'p-value': p_values})
            importance_df = importance_df.sort_values(by='Chi2', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Chi2', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)    
        elif m == 'ANOVA F-value':
            f_values, p_values = f_classif(X_encoded, y_encoded)
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'F-Value': f_values, 'p-value': p_values})
            importance_df = importance_df.sort_values(by='F-Value', ascending=False)
            
            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='F-Value', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)  
        elif m == 'Point Biserial Correlation':
            correlations = []
            for column in X_encoded.columns:
                corr, _ = pointbiserialr(y_encoded, X_encoded[column])
                correlations.append((column, corr))
            correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Point Biserial Correlation'])
            correlation_df = correlation_df.sort_values(by='Point Biserial Correlation', ascending=False)
            
            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Point Biserial Correlation', y='Feature', data=correlation_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)
            
        elif m == 'Mutual Information':
            mi_scores = mutual_info_classif(X_encoded, y_encoded)
            mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'Mutual Information': mi_scores})
            mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)

        else:
            model = RandomForestClassifier()
            model.fit(X_encoded, y_encoded)
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig = plt.figure(figsize=(10, 6))
            a = sns.barplot(x='Importance', y='Feature', data=importance_df)
            a = a.set_title("Feature Importance")
            st.pyplot(fig)

     

st.title("German Credit Risk Analysis and Modeling")

st.header("Data",divider=True)

df = pd.read_csv("German-Credit-Data-Analysis\Credit-Data-Raw.csv")
st.dataframe(df, use_container_width=True)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = [x for x in df.columns if x not in numerical_columns]

st.sidebar.header("Choose Tasks")
summary = st.sidebar.checkbox("Show Summary")
unique_values = st.sidebar.checkbox("Unique Values")
outlier_detection = st.sidebar.checkbox("Outlier Detection")
EDA = st.sidebar.checkbox("EDA")
feat_impt = st.sidebar.checkbox("Measure Feature Importance")

if summary:
    show_summary(df)

if unique_values:
    st.header("Unique Values:",divider=True)
    col_list_for_unique_vals = st.multiselect("Select Columns for Displaying Unique Values", options=df.columns)
    if col_list_for_unique_vals:
        show_unique_values(df,col_list_for_unique_vals)
    else:
        st.write("Please select at least one column to display its details.")

if outlier_detection:
    st.header("Outlier Detection:",divider=True)
    
    numerical_columns_stats = st.checkbox("Show Statistics of Numerical Attributes by target class for Outlier Detection")
    if numerical_columns_stats:
        show_outlier_detection(df,[],0)
    
    numerical_columns_viz = st.checkbox("Visualize Numerical Attributes for Outlier Detection")
    if numerical_columns_viz:
        col_list_for_otl_detection = st.multiselect("Select Attributes for visualization", options=numerical_columns)
        if col_list_for_otl_detection:
            show_outlier_detection(df,col_list_for_otl_detection,1)
        else:
            st.write("Please select at least one column to visualize.")
    

if EDA:
    st.header("Exploratory Data Analysis:", divider=True)
    univar_analysis = st.checkbox("Show Univariate Analysis")
    if univar_analysis:
        show_EDA(df,categorical_columns,0)
    
    bivar_analysis = st.checkbox("Show Bivariate Analysis")
    if bivar_analysis:
        show_EDA(df,df.columns,1)
        

    mulvar_analysis = st.checkbox("Show Multivariate Analysis")
    if mulvar_analysis:
        show_EDA(df,numerical_columns,2)

if feat_impt:
    st.header("Feature Importance", divider=True)
    num_feat_impt = st.checkbox("Measure Feature Importance of Numerical Attributes")
    if num_feat_impt:
        show_feat_impt(df,numerical_columns,0)
    cat_feat_impt = st.checkbox("Measure Feature Importance of Categorical Attributes")
    if cat_feat_impt:
        show_feat_impt(df,categorical_columns,1)


