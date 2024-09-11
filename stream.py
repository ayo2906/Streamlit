import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pickle


st.set_page_config(page_title="Bank Marketing Analysis", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    df = pd.concat([X, y], axis=1)
    return df

df = load_data()

#sidebar
st.sidebar.title("Bank Marketing Analysis")
st.sidebar.markdown("Use this app to explore the Bank Marketing dataset and build a predictive model.")
st.sidebar.markdown("---")


job_options = st.sidebar.multiselect('Select Job Types', df['job'].unique())
education_options = st.sidebar.multiselect('Select Education Levels', df['education'].unique())
marital_options = st.sidebar.multiselect('Select Marital Status', df['marital'].unique())


filtered_df = df.copy()
if job_options:
    filtered_df = filtered_df[filtered_df['job'].isin(job_options)]
if education_options:
    filtered_df = filtered_df[filtered_df['education'].isin(education_options)]
if marital_options:
    filtered_df = filtered_df[filtered_df['marital'].isin(marital_options)]

#tabs and layouts for data
tab1, tab2, tab3 = st.tabs(["Visualizations", "Exploratory Data Analysis" , "Predictive Modeling"])

#data visualization
with tab1:
    st.header("Data Visualizations")
    

    st.write("This section provides a comprehensive visual analysis of the Bank Marketing dataset, which is centered around a direct marketing campaign conducted by a Portuguese banking institution. The goal of the campaign was to predict whether a customer would subscribe to a term deposit based on a variety of factors such as age, job type, education level, marital status, and previous interactions with the bank. The visualizations here aim to uncover patterns and relationships within the data, allowing us to understand the underlying trends that influence customers' decisions.")
    #st.write("Age Distribution: An overview of how age is distributed among the customers.")
    #st.write("Job Distribution: Insights into the various job types held by the customers.")
    #st.write("Education Level Distribution: A pie chart representing the proportion of different education levels.")
    #st.write("Term Deposit by Marital Status: A comparison of term deposit subscriptions across different marital statuses.")
    #st.write("Box Plots: These help to compare the distributions of numeric variables across categories.")
    


    st.subheader("Term Deposits by Age and Job Type")
    fig = px.scatter(filtered_df, x='age', y='duration', color='job', size='campaign', hover_data=['education'])
    st.plotly_chart(fig)
    st.write("This scatter plot visualizes the relationship between customers' age and the duration of their last contact (phone call) with the bank. Each point on the plot represents a customer. The points are colored by job type, which helps to see how different professions are distributed across age and contact duration.")

    st.subheader("Distribution of Term Deposit Subscription")
    fig = px.histogram(filtered_df, x='y', color='y', title='Distribution of Term Deposit Subscription')
    st.plotly_chart(fig)
    st.write("This histogram visualizes the distribution of customers who subscribed (yes) versus those who did not subscribe (no) to a term deposit. It provides a clear view of the balance between the two outcomes in the dataset, helping to identify any imbalance in the target variable, which is important for predictive modeling.")


    st.subheader("Job Distribution")
    fig = px.histogram(filtered_df, x='job', title='Distribution of Job Types')
    st.plotly_chart(fig)
    st.write("This histogram represents the distribution of different job types among customers. It shows which professions are more common in the dataset, providing insight into the socio-economic status of the customer base.")

    # education Level Pie Chart
    st.subheader("Education Level Distribution")
    fig = px.pie(filtered_df, names='education', title='Proportion of Education Levels')
    st.plotly_chart(fig)
    st.write("This pie chart displays the proportion of customers with different levels of education. It gives a quick overview of the educational background of the customer base, which could influence their financial decisions.")

    #term Deposit by Marital Status
    st.subheader("Term Deposit by Marital Status")
    fig = px.histogram(filtered_df, x='marital', color='y', barmode='group', title='Term Deposit by Marital Status')
    st.plotly_chart(fig)
    st.write("This bar chart compares the subscription to term deposits across different marital statuses. It helps understand how marital status might affect a customer's likelihood of subscribing to a term deposit.")


    # Feature Correlation with Target
    #st.subheader("Feature Correlation with Target")
    #target_corr = corr_matrix['y_yes'].sort_values(ascending=False)
    #st.bar_chart(target_corr)


    # box Plot for Numeric Variables
    st.subheader("Box Plot for Numeric Variables")
    numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
    selected_column = st.selectbox("Select a numeric column", numeric_columns)
    fig = px.box(filtered_df, x='y', y=selected_column, title=f'Box Plot of {selected_column}')
    st.plotly_chart(fig)
    st.write("This box plot shows the distribution of a selected numeric variable across customers who did or did not subscribe to a term deposit. It provides insights into the spread and outliers of the data for that particular variable.")

    # Pair Plot (Seaborn)
    st.subheader("Pair Plot")
    sns.pairplot(filtered_df, hue='y', diag_kind='kde')
    st.pyplot(plt)
    st.write("A pair plot is a grid of scatter plots and histograms that visualizes relationships between multiple pairs of variables in the dataset. Each scatter plot represents the relationship between two features, while the diagonal plots show the distribution of individual features (in this case, using kernel density estimates). ")

    # Histogram of Balance
    st.subheader("Histogram of Balance")
    fig = px.histogram(filtered_df, x='balance', nbins=50, title='Distribution of Customer Balance')
    st.plotly_chart(fig)
    st.write("This histogram visualizes the distribution of customer balances in the dataset. By showing how many customers fall into different balance ranges, it provides insights into the financial status of the customer base. The number of bins (set to 50) determines the granularity of the distribution, allowing you to see how balances are spread out, whether most customers have low balances, or if there's a significant number of high-balance customers. This can help identify target groups for marketing efforts or assess the overall financial health of the customer base.")

    # Box Plot of Duration by Job
    st.subheader("Box Plot of Duration by Job")
    fig = px.box(filtered_df, x='job', y='duration', title='Duration of Last Contact by Job Type')
    st.plotly_chart(fig)
    st.write("This box plot visualizes the distribution of the duration of the last contact (in seconds) by different job types. Each box represents the range of contact durations for a specific job type, with the line inside the box indicating the median duration. This visualization helps to identify how contact durations vary across different professions, and whether certain job types tend to have longer or shorter conversations during marketing calls. It can also highlight any outliers, such as particularly long or short calls within each job category.")
    
    # Term Deposit Subscription by Age Group
    st.subheader("Term Deposit Subscription by Age Group")
    age_bins = pd.cut(filtered_df['age'], bins=[18, 30, 40, 50, 60, 100])
    age_grouped = filtered_df.groupby(age_bins)['y'].value_counts(normalize=True).unstack().fillna(0)
    fig = age_grouped.plot(kind='bar', stacked=True, figsize=(10, 6), title='Term Deposit Subscription by Age Group')
    st.pyplot(fig.get_figure())
    st.write("This bar chart shows the proportion of customers in different age groups who either subscribed or did not subscribe to a term deposit. The age groups are defined by custom bins (e.g., 18-30, 31-40, etc.). The chart is stacked to show the relative percentage of subscriptions and non-subscriptions within each age group. This visualization helps identify whether certain age groups are more likely to subscribe to a term deposit, offering insights into the age-related preferences of the customer base.")


    # Scatter Plot of Balance vs. Age
    st.subheader("Scatter Plot of Balance vs. Age")
    fig = px.scatter(filtered_df, x='age', y='balance', color='y', title='Customer Balance vs. Age')
    st.plotly_chart(fig)
    st.write("This scatter plot explores the relationship between customer balance and age. By coloring the points according to whether the customer subscribed to a term deposit, it helps visualize patterns or trends, such as whether older customers tend to have higher balances.")

    # Violin Plot of Duration by Education Level
    st.subheader("Violin Plot of Duration by Education Level")
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='education', y='duration', hue='y', data=filtered_df, split=True, inner='quart')
    plt.title('Distribution of Contact Duration by Education Level')
    st.pyplot(fig)
    st.write("This violin plot shows the distribution of contact duration (the length of the last phone call) across different education levels. It combines elements of box plots and kernel density plots to provide a detailed view of the data distribution, especially useful for comparing multiple groups.")

   
    
#eda
with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Get an overview of the dataset and understand its basic characteristics.")
    st.write("### Dataset Overview")
    st.dataframe(filtered_df.head(10))

    st.write("### Descriptive Statistics")
    st.write(filtered_df.describe())

    st.write("### Missing Values")
    st.write(filtered_df.isnull().sum())

#all nums with heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = filtered_df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write("This heatmap shows the correlation between numeric features in the dataset. A correlation value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation. This helps identify relationships between variables, such as how age or balance might relate to term deposit subscriptions.")

#predictive modelling
with tab3:
    st.header("Predictive Modeling")
    st.write("### Build a Random Forest Classifier")

    df_model = filtered_df.copy()
    df_model = pd.get_dummies(df_model, drop_first=True)  

    X = df_model.drop('y_yes', axis=1)
    y = df_model['y_yes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    #user_input = st.text_input("Job", job_options)

#performance of the set in classifying the test data.
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.write("### Feature Importance")
    st.write("This feature graph presentation displays the importance of each dataset in determining the outcome for each customer.")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importances)

    st.write("### Predict New Data")

    with st.form(key='user_input_form'):
        st.write("### Enter your details:")

        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        job = st.selectbox("Job", options=['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", options=['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Education Level", options=['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.selectbox("Credit in Default", options=['yes', 'no', 'unknown'])
        balance = st.number_input("Balance in Euros", min_value=0, step=10)
        housing = st.selectbox("Housing Loan?", options=['yes', 'no', 'unknown'])
        loan = st.selectbox("Personal Loan?", options=['yes', 'no', 'unknown'])
        #contact = st.selectbox("Contact Communication", options=['mobile', 'cellular', 'unknown'])
        contact = st.selectbox("Contact Communication", options=['telephone','cellular','unknown'])
        day = st.slider("Day of the Month(Previous contact)", min_value=1, max_value=31, step=1)
        month = st.selectbox("Last Contact Month", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        duration = st.number_input("Duration of last contact(in seconds)", min_value=0, step=10)
        campaign = st.number_input("Number of Contacts performed during Campaign", min_value=1, step=1)
        pdays = st.number_input("Number of Days Since Last Contact(pdays)", min_value=-1, step=1)
        previous = st.number_input("Number of Contacts before this campaign", min_value=0, step=1)
        #poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=['success', 'failure', 'other', 'unknown'])
        poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=['success', 'failure', 'other','unknown'])

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        user_input_data = {
            'age' : [age],
            'job' : [job],
            'marital' : [marital],
            'education' : [education],
            'default' : [default],
            'balance' : [balance],
            'housing' : [housing],
            'loan' : [loan],
            #'contact' : [contact],
            'day' : [day],
            'month' : [month],
            'duration' : [duration],
            'campaign' : [campaign],
            'pdays' : [pdays],
            'previous' : [previous],
            #'poutcome' : [poutcome]
        }

        user_df = pd.DataFrame(user_input_data)
        user_df_encoded = pd.get_dummies(user_df).reindex(columns=X.columns,fill_value=0)
        prediction = model.predict(user_df_encoded)
        #predict_probability = model.predict_proba(user_df_encoded)
        result = "Subscribed to Term Deposit" if prediction[0] == 1 else "Did Not Subscribe to Term Deposit"
        st.subheader(f"Prediction : {result}")
        #st.subheader(f"Pobability of prediction: {predict_probability}")


#page style
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp {background-color: black;}
    </style>
""", unsafe_allow_html=True)

def load_components_function(fp):
    
    with open(fp,"rb") as f:
        object = pickle.load(f)
    return object




