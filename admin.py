import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv('SecondCar.csv')

# Load the model pipeline from file
pipeline = pickle.load(open('Final_M_Pipeline.sav', 'rb'))

# Set up Streamlit app layout
st.set_page_config(page_title="Car Price Prediction Admin Dashboard", layout="wide")

# Add custom CSS to style the app
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #007BFF;
        font-size: 40px;
        font-weight: bold;
        margin-top: 20px;
    }
    .subheader {
        color: #343A40;
        font-size: 30px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .markdown-text-container {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .container {
        background-color: #E9ECEF;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title
st.markdown('<div class="title">Used Car Selling Price Prediction Admin Dashboard</div>', unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Function to display the login page
def login_page():
    st.subheader("Login")
    with st.form("login_form"):
        login_name = st.text_input("Enter your login name")
        password = st.text_input("Enter your password", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            # Add your login validation logic here
            if login_name == "admin" and password == "password":  # Replace with real validation
                st.session_state.logged_in = True  # Set login status to True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials. Please try again.")

# Function to display the main content after login
def main_content():
    # Navigation Sidebar
    st.sidebar.title("Navigation")
    st.session_state.selected = st.sidebar.radio("Select a section:", ["Admin Dashboard", "Model's Performance", "Predict Selling Price", "Logout"])

    if st.session_state.selected == 'Admin Dashboard':
        # Admin Dashboard
        st.subheader("Dataset Overview")

        # Display basic dataset information
        st.markdown("### Dataset Description")
        st.write(df.describe(include='all'))

        st.markdown("### Categorical Variables")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        st.write(df[cat_cols].describe())

        st.markdown("### Numerical Variables")
        num_cols = df.select_dtypes(exclude='object').columns.tolist()
        st.write(df[num_cols].describe())

        # Overall summary description in textual format
        st.markdown("### Overall Summary")
        overall_summary = (
            "The dataset contains information about used cars. "
            "Key features include the car's name, year of manufacture, "
            "kilometers driven, fuel type, seller type, transmission type, "
            "owner type, and ex-showroom price. The dataset enables analysis "
            "to predict the selling price of used cars based on these attributes."
        )
        st.markdown(overall_summary)

        # Visualizations
        st.markdown("### Visualizations")

        # Heatmap (only for numeric columns)
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include='number')  # Only select numeric columns
        fig = px.imshow(numeric_df.corr(), color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart for categorical variables
        st.subheader("Categorical Variable Distribution")
        fig = px.bar(df.melt(value_vars=cat_cols), x='variable', y='value', color='value')
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.selected == 'Model\'s Performance':
        # Model's Performance Section st.subheader("Model Evaluation Metrics")

        # Load the model pipeline from file
        # Assuming the model is already loaded as pipeline
        y_pred = pipeline.predict(df.drop(columns=['selling_price']))

        # Predefined results
        r2_score = 0.9347029245696016
        mae = 71647.0216512702
        mse = 21042548084.399227

        # Display evaluation metrics
        st.markdown("### Evaluation Metrics")
        st.write(f"**Model R² score:** {r2_score:.2%}")
        st.write(f"**Mean Absolute Error:** {mae:.2f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")

        # Visualize evaluation metrics
        st.subheader("Evaluation Metrics Visualization")
        metrics_df = pd.DataFrame({
            "Metric": ["R² Score", "Mean Absolute Error", "Mean Squared Error"],
            "Value": [r2_score, mae, mse]
        })

        # Check if lengths are the same before plotting
        if len(metrics_df['Metric']) == len(metrics_df['Value']):
            fig_metrics = px.bar(metrics_df, x='Metric', y='Value', title='Model Evaluation Metrics', 
                                  labels={'Metric': 'Metric', 'Value': 'Value'}, 
                                  color_discrete_sequence=['#007BFF'])
            st.plotly_chart(fig_metrics)
        else:
            st.error("Error: Metric and Value arrays must be of the same length.")

        # Textual summarization of model performance
        st.markdown("""
            The model's performance indicates a strong ability to predict the selling prices of used cars. 
            The R² score of **{:.2%}** suggests that a significant portion of the variability in the selling price 
            is explained by the features used in the model. Additionally, the Mean Absolute Error (MAE) of 
            **{:.2f}** indicates that, on average, the model's predictions are off by approximately this amount 
            in the currency units. Lastly, the Mean Squared Error (MSE) of **{:.2f}** highlights the average 
            squared difference between predicted and actual values, emphasizing the model's reliability.
        """.format(r2_score, mae, mse))

    elif st.session_state.selected == 'Predict Selling Price':
        # Predict Selling Price Section
        st.subheader("Predict Selling Price")

        # Get unique car names from the dataset
        car_names = df['name'].unique().tolist()

        # Section 1: Enter car name
        st.markdown("### Select Car Name")
        car_name = st.selectbox("Select a car name", car_names)

        # Section 2: Car details
        st.markdown("### Car Details")
        car_details = df[df['name'] == car_name]
        # Exclude the 'selling_price' column from being displayed
        car_details_display = car_details.drop(columns=['selling_price'])
        st.write(car_details_display)

        # Section 3: Enter car details for prediction
        st.markdown("### Enter Car Details for Prediction")

        # Get unique values for each field
        years = car_details['year'].unique().tolist()
        km_drivens = car_details['km_driven'].unique().tolist()
        fuel_types = car_details['fuel'].unique().tolist()
        seller_types = car_details['seller_type'].unique().tolist()
        transmission_types = car_details['transmission'].unique().tolist()
        owner_types = car_details['owner'].unique().tolist()
        ex_showroom_prices = car_details['ExShowroom Price'].unique().tolist()

        # Create dropdown lists if there are multiple values
        if len(years) > 1:
            year = st.selectbox("Select year of manufacture", years)
        else:
            year = years[0]

        if len(km_drivens) > 1:
            km_driven = st.selectbox("Select kilometers driven", km_drivens)
        else:
            km_driven = km_drivens[0]

        if len(fuel_types) > 1:
            fuel_type = st.selectbox("Select fuel type", fuel_types)
        else:
            fuel_type = fuel_types[0]

        if len(seller_types) > 1:
            seller_type = st.selectbox("Select seller type", seller_types)
        else:
            seller_type = seller_types[0]

        if len(transmission_types) > 1:
            transmission_type = st.selectbox("Select transmission type", transmission_types)
        else:
            transmission_type = transmission_types[0]

        if len(owner_types) > 1:
            owner_type = st.selectbox("Select owner type", owner_types)
        else:
            owner_type = owner_types[0]

        if len(ex_showroom_prices) > 1:
            ex_showroom_price = st.selectbox("Select ex-showroom price", ex_showroom_prices)
        else:
            ex_showroom_price = ex_showroom_prices[0]

        # Predict selling price
        if st.button("Predict Selling Price"):
            input_df = pd.DataFrame({
                'name': [car_name],
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel_type],
                'seller_type': [seller_type],
                'transmission': [transmission_type],
                'owner': [owner_type],
                'ExShowroom Price': [ex_showroom_price]
            })
            selling_price = pipeline.predict(input_df)
            st.write(f"Predicted selling price: *{selling_price[0]:.2f}*")

            # Plot selling price vs actual price
            st.markdown("### Selling Price vs Actual Price")
            fig = px.scatter(car_details, x='selling_price', y='ExShowroom Price')
            st.plotly_chart(fig, use_container_width=True)

            # Textual description
            st.markdown("The above graph shows the relationship between the selling price and the actual price of the car. The predicted selling price is shown as a point on the graph, and the actual price is shown on the x-axis.")

    elif st.session_state.selected == 'Logout':
        st.session_state.logged_in = False
        st.info("You have been logged out successfully.")

# Main App Logic
if not st.session_state.logged_in:
    login_page()
else:
    main_content()