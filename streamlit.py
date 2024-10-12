import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

# Load the dataset
df = pd.read_csv('SecondCar.csv')

# Load the model pipeline from file
pipeline = pickle.load(open('Final_Model_Pipeline.sav', 'rb'))

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
            
            # Select km driven from dropdown based on selected car
            km_driven_options = car_data['km_driven'].unique()
            km_driven = st.selectbox("Select Kilometers Driven", km_driven_options)
            
            # Select ExShowroom Price from dropdown based on selected car
            exshowroom_price_options = car_data['ExShowroom Price'].unique()
            exshowroom_price = st.selectbox("Select ExShowroom Price", exshowroom_price_options)

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
            st.markdown(f'<h3 style="color: #FF5733;">Predicted Selling Price: ₹{round(predicted_price[0], 2)}</h3>', unsafe_allow_html=True)
            
            # Create a container for visualizations
            with st.container():
                st.subheader("Visualizations")
                st.markdown('<div class="container">', unsafe_allow_html=True)

                # Predicted vs Actual Selling Prices for the selected car
                car_sales_data = df[df['name'] == car_name]
                car_sales_data['Predicted Price'] = pipeline.predict(car_sales_data)
                
                # 3D Scatter Plot (Selling Price vs ExShowroom Price and Km Driven)
                st.subheader("3D Scatter Plot: Selling Price vs ExShowroom Price vs Km Driven")
                fig_scatter = go.Figure(data=[go.Scatter3d(
                    x=car_sales_data['ExShowroom Price'],
                    y=car_sales_data['km_driven'],
                    z=car_sales_data['selling_price'],
                    mode='markers',
                    marker=dict(size=5, color=car_sales_data['selling_price'], colorscale='YlGnBu', opacity=0.8),
                    text=car_sales_data['name'],  # Hover text
                )])
                fig_scatter.update_layout(title="3D Scatter Plot: Selling Price vs ExShowroom Price vs Km Driven",
                                           scene=dict(
                                               xaxis_title='ExShowroom Price',
                                               yaxis_title='Km Driven',
                                               zaxis_title='Selling Price'))
                st.plotly_chart(fig_scatter)

                st.markdown("""
                    **3D Scatter Plot:** This plot visualizes the relationship between the ExShowroom Price, Kilometers Driven, and Selling Price. 
                    Each point represents a car, with its position determined by its ExShowroom Price and Km Driven, while the height indicates the Selling Price.
                """)

                # 3D Surface Plot (Selling Price over ExShowroom Price and Km Driven)
                st.subheader("3D Surface Plot: Selling Price over ExShowroom Price and Km Driven")
                grid_x, grid_y = np.meshgrid(car_sales_data['ExShowroom Price'], car_sales_data['km_driven'])
                grid_z = griddata(
                    (car_sales_data['ExShowroom Price'], car_sales_data['km_driven']),
                    car_sales_data['selling_price'],
                    (grid_x, grid_y),
                    method='linear'
                )

                fig_surface = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
                fig_surface.update_layout(title="3D Surface Plot: Selling Price over ExShowroom Price and Km Driven",
                                           scene=dict(
                                               xaxis_title='ExShowroom Price',
                                               yaxis_title='Km Driven',
                                               zaxis_title='Selling Price'))
                st.plotly_chart(fig_surface)

                st.markdown("""
                    **3D Surface Plot:** This surface plot provides a smooth visualization of how the Selling Price varies 
                    with respect to both the ExShowroom Price and Kilometers Driven. It helps in understanding the trends in selling prices.
                """)

                # 3D Scatter Plot (Predicted vs Actual Prices for the Selected Car)
                st.subheader("3D Scatter Plot: Actual vs Predicted Selling Price")
                fig_pred_actual = go.Figure(data=[go.Scatter3d(
                    x=car_sales_data['year'],
                    y=car_sales_data['selling_price'],
                    z=car_sales_data['Predicted Price'],
                    mode='markers',
                    marker=dict(size=5, color='orange', opacity=0.8),
                )])
                fig_pred_actual.update_layout(title="3D Scatter Plot: Actual vs Predicted Selling Price",
                                              scene=dict(
                                                  xaxis_title='Year',
                                                  yaxis_title='Actual Selling Price',
                                                  zaxis_title='Predicted Selling Price'))
                st.plotly_chart(fig_pred_actual)

                st.markdown("""
                    **3D Scatter Plot:** This plot compares the actual and predicted selling prices over different years for the selected car.
                    It serves to evaluate the accuracy of the predictions made by the model.
                """)

                st.markdown('</div>', unsafe_allow_html=True)

