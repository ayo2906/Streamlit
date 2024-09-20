**Bank Marketing Analysis - Streamlit Application**

**Overview**

This project is a Streamlit-based application that allows users to explore the Bank Marketing Dataset and build predictive models for term deposit subscriptions. The application provides multiple data visualizations, descriptive statistics, and a machine learning model to predict whether a customer will subscribe to a term deposit based on various input features.

**Features**

1. Data Exploration: Users can explore the dataset by filtering it based on job type, education level, and marital status. The application provides several interactive visualizations to help users understand the data better.

2. Data Visualizations: The app includes various visualizations such as:

- Correlation Heatmap

- Term Deposit Subscription by Age Group

- Age Distribution

- Job and Education Distribution

- Term Deposit by Marital Status

- Scatter and Box Plots for various relationships

3. Exploratory Data Analysis (EDA): The EDA section provides:

- Dataset overview

- Descriptive statistics

- Missing values summary

4. Predictive Modeling: The application allows users to:

- Train a Random Forest Classifier on the dataset

- View the classification report and confusion matrix

- Analyze feature importance

- Predict whether a new customer will subscribe to a term deposit based on input details

**How the Application Works**

**1. Loading Data**

The data is fetched from the UCI Machine Learning Repository using the fetch_ucirepo package. It includes features related to customers' demographic, financial, and contact information.

**2. Sidebar Filters**

Users can filter the dataset based on:

- Job Types

- Education Levels

- Marital Status
  
**3. Tabs for Data Exploration**

The app is divided into three main tabs:

- Visualizations Tab: Displays a series of visualizations to explore different aspects of the data.

  - Example: The Correlation Heatmap shows the relationship between numerical variables in the dataset, while the Term Deposit Subscription by Age Group breaks down the likelihood of subscription by age category.

- Exploratory Data Analysis (EDA) Tab: Provides a summary of the dataset, including descriptive statistics and missing values.

- Predictive Modeling Tab: In this section, users can:

  - Train a Random Forest Classifier to predict term deposit subscriptions.
    
  - View the model’s performance via a confusion matrix and classification report.
    
  - Input new customer details and make predictions on whether they will subscribe to a term deposit.
    
**4. Random Forest Classifier**

- The Random Forest model is trained using customer data (after encoding categorical variables).

- After training, the application displays feature importance, helping users understand which variables most influence term deposit subscriptions.

**5. New Predictions**

Users can enter new customer information through an input form, and the app will predict whether the customer will subscribe to a term deposit based on the trained model.

**Instructions for Running the Application**

1. Install the required packages: Ensure you have the following Python packages installed:

pip install streamlit pandas seaborn plotly matplotlib scikit-learn ucimlrepo

2. Run the Streamlit application: Navigate to the project directory and run the following command in your terminal:

streamlit run app.py

3. Explore the Application:

- Use the sidebar to filter data.

- Navigate through the tabs to explore visualizations, EDA, and predictive modeling.

- Enter new customer data for prediction and view the result.

**Key Visualizations**

1. Correlation Heatmap

Shows relationships between numeric variables, indicating how features such as balance or age relate to term deposit subscription.

2. Term Deposit Subscription by Age Group

Bar chart representing the proportion of customers in various age groups who subscribed to a term deposit.

3. Job Distribution

A bar chart that shows the distribution of job types across the dataset.

4. Term Deposit by Marital Status

A bar chart that compares the likelihood of subscribing based on marital status.

**Future Enhancements**

- Add more advanced machine learning models for comparison.

- Improve the user interface for a better user experience.

- Include additional filters for more granular analysis.

**Conclusion**

This Streamlit application allows users to explore the Bank Marketing Dataset interactively and gain insights through visualizations and predictive modeling. It provides a simple yet effective way to analyze customer behavior and predict future term deposit subscriptions.










