import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os

# Load the dataset
df = pd.read_csv('SecondCar.csv')

# Load the model pipeline from file
pipeline = pickle.load(open('Final_M_Pipeline.sav', 'rb'))

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

# Streamlit App
st.markdown('<div class="title">Used Car Selling Price Prediction</div>', unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login Page
if not st.session_state.logged_in:
    st.subheader("Login")
    with st.form("login_form"):
        login_name = st.text_input("Enter your login name")
        password = st.text_input("Enter your password", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state.logged_in = True  # Set login status to True
            st.success("Login successful!")

# If logged in, display the main app
if st.session_state.logged_in:
    # Select Car Option
    st.subheader("Select Car Option")
    car_name = st.selectbox("Select your car name", df['name'].unique())
    
    if car_name:
        # Get the relevant data for the selected car
        car_data = df[df['name'] == car_name]
        
        with st.expander("Select Car Features", expanded=True):
            # Select Fuel Type
            fuel_type = st.selectbox("Select your fuel type", car_data['fuel'].unique())
            
            # Select Seller Type
            seller_type = st.selectbox("Select your seller type", car_data['seller_type'].unique())
            
            # Select Transmission Type
            transmission_type = st.selectbox("Select your transmission type", car_data['transmission'].unique())
            
            # Select Owner Type
            owner_type = st.selectbox("Select your owner type", car_data['owner'].unique())
            
            # Select Year from dropdown based on selected car
            year_options = car_data['year'].unique()
            year = st.selectbox("Select Year", year_options)
            
            # Enter kilometers driven (allow free input)
            km_driven = st.number_input("Enter Kilometers Driven", min_value=0.0, max_value=1000000.0, value=float(car_data['km_driven'].mean()), step=0.1)
            
            # Enter any ExShowroom Price (allow free input)
            exshowroom_price = st.number_input("Enter ExShowroom Price", min_value=0.0, max_value=5000000.0, value=float(car_data['ExShowroom Price'].mean()), step=0.1)

        # Create a button to predict the selling price
        if st.button("Predict Selling Price"):
            # Create a DataFrame with the selected options
            input_data = pd.DataFrame({
                'name': [car_name],
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel_type],
                'seller_type': [seller_type],
                'transmission': [transmission_type],
                'owner': [owner_type],
                'ExShowroom Price': [exshowroom_price]
            })
            
            # Predict selling price
            predicted_price = pipeline.predict(input_data)
            st.markdown(f'<h3 style="color: #39FF14;">Predicted Selling Price: ₹{round(predicted_price[0], 2)}</h3>', unsafe_allow_html=True)
            
            # Create a container for visualizations
            with st.container():
                st.subheader("Visualizations")
                st.markdown('<div class="container">', unsafe_allow_html=True)

                # 2D Scatter Plot: ExShowroom Price vs Selling Price
                st.subheader("Scatter Plot: ExShowroom Price vs Selling Price")
                fig_scatter = px.scatter(df, x='ExShowroom Price', y='selling_price', color='fuel', title="ExShowroom Price vs Selling Price",
                                         labels={'ExShowroom Price': 'ExShowroom Price', 'selling_price': 'Selling Price'})
                st.plotly_chart(fig_scatter)

                # Save scatter plot
                if st.button("Download Scatter Plot"):
                    pio.write_image(fig_scatter, "scatter_plot.png")
                    st.success("Scatter plot saved as 'scatter_plot.png'")

                # 2D Line Plot: Year vs Selling Price
                st.subheader("Line Plot: Year vs Selling Price")
                fig_line = px.line(df, x='year', y='selling_price', color='seller_type', title="Year vs Selling Price",
                                   labels={'year': 'Year', 'selling_price': 'Selling Price'})
                st.plotly_chart(fig_line)

                # Save line plot
                if st.button("Download Line Plot"):
                    pio.write_image(fig_line, "line_plot.png")
                    st.success("Line plot saved as 'line_plot.png'")

                # Bar Plot: Fuel Type vs Average Selling Price
                st.subheader("Bar Plot: Fuel Type vs Average Selling Price")
                fuel_avg_price = df.groupby('fuel')['selling_price'].mean().reset_index()
                fig_bar = px.bar(fuel_avg_price, x='fuel', y='selling_price', title="Fuel Type vs Average Selling Price",
                                 labels={'fuel': 'Fuel Type', 'selling_price': 'Average Selling Price'})
                st.plotly_chart(fig_bar)

                # Save bar plot
                if st.button("Download Bar Plot"):
                    pio.write_image(fig_bar, "bar_plot.png")
                    st.success("Bar plot saved as 'bar_plot.png'")

                # Predicted vs Actual Selling Prices for the selected car
                car_sales_data = df[df['name'] == car_name]
                car_sales_data['Predicted Price'] = pipeline.predict(car_sales_data)

                # 2D Scatter Plot: Actual vs Predicted Selling Price
                st.subheader("Scatter Plot: Actual vs Predicted Selling Price")
                fig_pred_actual = px.scatter(car_sales_data, x='selling_price', y='Predicted Price', color='year',
                                             title="Actual vs Predicted Selling Price",
                                             labels={'selling_price': 'Actual Selling Price', 'Predicted Price': 'Predicted Selling Price'})
                st.plotly_chart(fig_pred_actual)

                # Save predicted vs actual plot
                if st.button("Download Actual vs Predicted Plot"):
                    pio.write_image(fig_pred_actual, "actual_vs_predicted.png")
                    st.success("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

                st.markdown('</div>', unsafe_allow_html=True)
