import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

            

st.title("German Credit Risk Analysis and Modeling")

st.header("Data",divider=True)
df = pd.read_csv("Credit-Data-Raw.csv")
st.dataframe(df, use_container_width=True)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = [x for x in df.columns if x not in numerical_columns]

st.sidebar.header("Choose Tasks")
summary = st.sidebar.checkbox("Show Summary")
unique_values = st.sidebar.checkbox("Unique Values")
outlier_detection = st.sidebar.checkbox("Outlier Detection")
EDA = st.sidebar.checkbox("EDA")


if summary:
    show_summary(df)

if unique_values:
    st.header("Unique Values:",divider=True)
    col_list_for_unique_vals = st.multiselect("Select Columns for Displaying Unique Values", options=df.columns)
    if col_list_for_unique_vals:
        show_unique_values(df,col_list_for_unique_vals)
    else:
        st.write("Please select at least one column to display its details.")
    st.success("Findings:")
    st.write("1) The Elements in each column are consistent with their respective column's datatype")
    st.write("2) The column 'credit_history' contains some values eqaul to 'no credits/all paid' and some values equal to 'all paid'. So Values equal to 'no credits/all paid' needs to be replaced to 'no credits' in 'credit_history' column to avoid confusion and maintain uniqueness of each sample.")
    st.write("3) The column 'personal status' can be divided into two columns: one for gender and other for marital status.")

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
    
    st.success("Observations:")
    st.write("1) For duration: 75% of the population with bad credit is having duration of 36 months or less whereas same percent of those with good credit are having duration of 24 months or less.")
    st.write("2) For credit_amount: 75% of the population with bad credit is having credit_amount of 5141.5 or less whereas same percent of those with good credit are having credit_amount of 3634.75 or less.")
    st.write("3) Hence we cannot eleminate samples as these two columns are critical to the target class.")

if EDA:
    st.header("Exploratory Data Analysis:", divider=True)
    univar_analysis = st.checkbox("Show Univariate Analysis")
    if univar_analysis:
        show_EDA(df,categorical_columns,0)
    
    bivar_analysis = st.checkbox("Show Bivariate Analysis")
    if bivar_analysis:
        show_EDA(df,df.columns,1)