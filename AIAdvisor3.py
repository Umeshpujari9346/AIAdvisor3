import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from keras.saving import register_keras_serializable
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from PIL import Image
import io

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set Streamlit page configuration
st.set_page_config(page_title="AI-Driven Smart Agriculture Advisor", layout="wide")

# Custom loss function for LSTM model
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load LSTM model with caching
@st.cache_resource
def load_lstm_model():
    try:
        model = tf.keras.models.load_model("lstm_model_fixed.h5", custom_objects={"mse": mse}, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading LSTM model: {e}")
        return None

# Load and preprocess dataset
df = pd.read_csv("data_season.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Handle missing values in categorical columns
categorical_cols = ["soil_type", "location", "crops", "season", "irrigation"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Load machine learning models with caching
@st.cache_resource
def load_ml_models():
    models = {}

    # Existing yield prediction model
    yield_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
    yield_model.fit(df[["rainfall", "temperature", "soil_type", "irrigation", "humidity", "area"]], df["yeilds"])
    models["yield"] = yield_model

    # Price prediction model
    price_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    price_model.fit(df[["year", "location", "crops", "yeilds", "season"]], df["price"])
    models["price"] = price_model

    # ARIMA model for price forecasting
    arima_model = auto_arima(df["price"], seasonal=True, stepwise=True, suppress_warnings=True)
    models["arima"] = arima_model

    # Crop selection model
    crop_model = XGBClassifier(n_estimators=200, learning_rate=0.1)
    crop_model.fit(df[["rainfall", "temperature", "soil_type", "humidity", "season", "irrigation", "location"]], df["crops"])
    models["crop"] = crop_model

    # Irrigation recommendation model
    irrigation_model = RandomForestClassifier()
    irrigation_model.fit(df[["crops", "soil_type", "rainfall", "temperature"]], df["irrigation"])
    models["irrigation"] = irrigation_model

    # New yield prediction model including crops
    yield_with_crop_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
    yield_with_crop_model.fit(df[["season", "soil_type", "location", "crops", "rainfall", "temperature", "humidity", "irrigation", "area"]], df["yeilds"])
    models["yield_with_crop"] = yield_with_crop_model

    return models

# Load models
ml_models = load_ml_models()
lstm_model = load_lstm_model()

# Function to simulate IoT data
def get_iot_data():
    # Simulate soil moisture and weather data
    soil_moisture = np.random.uniform(10, 30)  # Percentage
    temperature = np.random.uniform(15, 35)  # Celsius
    humidity = np.random.uniform(40, 80)  # Percentage
    return soil_moisture, temperature, humidity

# Function to recommend irrigation
def recommend_irrigation(soil_moisture, temperature, humidity):
    if soil_moisture < 15:
        return "Irrigate now: 50L/ha"
    elif soil_moisture < 20 and temperature > 30:
        return "Irrigate soon: 30L/ha"
    else:
        return "No irrigation needed"

# Function to simulate community insights
def get_community_insights(location):
    location_df = df[df["location"] == location]
    if location_df.empty:
        return None
    top_crop = location_df["crops"].mode()[0]
    avg_yield = location_df[location_df["crops"] == top_crop]["yeilds"].mean()
    return label_encoders["crops"].inverse_transform([top_crop])[0], avg_yield

# Function to simulate climate trends
def get_climate_trend():
    # Simulate a decreasing trend in rainfall
    current_rainfall = df["rainfall"].mean()
    future_rainfall = current_rainfall * 0.9  # 10% decrease
    return current_rainfall, future_rainfall

# Function to recommend climate-resilient crops
def recommend_climate_resilient_crops(future_rainfall):
    # Simple rule: if future rainfall decreases, recommend drought-resistant crops
    if future_rainfall < df["rainfall"].mean():
        return "Consider drought-resistant crops like millet or sorghum."
    else:
        return "No significant climate impact predicted."

# Main application function
def main():
    st.title("🌾 AI-Driven Smart Agriculture Advisor")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Select Analysis Section", [
        "📊 Model Performance",
        "📈 Price Forecasts",
        "🤖 LSTM Predictions",
        "🌾 Optimal Crop Selection",
        "📉 XGBoost Predictions",
        "💧 Irrigation Recommendation",
        "🌱 Best Crop Prediction",
        "📸 Pest & Disease Scanner",
        "💧 Smart Irrigation",
        "🎲 Profit Simulator",
        "🌐 Community Insights",
        "🌍 Climate Planner"
    ])

    # **Model Performance Section**
    if selected_page == "📊 Model Performance":
        st.subheader("📊 Model Performance")
        test_size = int(0.2 * len(df))
        df_train, df_test = df[:-test_size], df[-test_size:]

        y_true_price = df_test["price"]
        y_pred_price = ml_models["price"].predict(df_test[["year", "location", "crops", "yeilds", "season"]])

        rmse_price = mean_squared_error(y_true_price, y_pred_price, squared=False)
        r2_price = r2_score(y_true_price, y_pred_price)

        st.metric("💰 Price Prediction RMSE", f"{rmse_price:.2f}", "Lower is better")
        st.metric("📊 R² Score (Price Model)", f"{r2_price:.2f}", "Closer to 1 is better")

        # Scatter plot for actual vs predicted prices
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_true_price, y=y_pred_price, ax=ax)
        ax.set_xlabel("Actual Price (₹)")
        ax.set_ylabel("Predicted Price (₹)")
        ax.set_title("Actual vs. Predicted Prices")
        max_val = max(y_true_price.max(), y_pred_price.max())
        ax.plot([0, max_val], [0, max_val], "k--", label="Perfect Prediction")
        ax.legend()
        st.pyplot(fig)

    # **Price Forecasts Section**
    elif selected_page == "📈 Price Forecasts":
        st.subheader("📈 Price Forecasts")
        crop = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        location = st.selectbox("Select Location", label_encoders["location"].classes_)
        season = st.selectbox("Select Season", label_encoders["season"].classes_)
        year = st.slider("Select Year", 2000, 2030, 2025)

        crop_encoded = label_encoders["crops"].transform([crop])[0]
        location_encoded = label_encoders["location"].transform([location])[0]
        season_encoded = label_encoders["season"].transform([season])[0]
        average_yield = df[df["crops"] == crop_encoded]["yeilds"].mean()

        predicted_price = ml_models["price"].predict([[year, location_encoded, crop_encoded, average_yield, season_encoded]])[0]
        st.success(f"Predicted Crop Price: ₹{predicted_price:.2f}")

        # Historical price trend
        historical_data = df[(df["crops"] == crop_encoded) & (df["location"] == location_encoded) & (df["season"] == season_encoded)]
        if not historical_data.empty:
            historical_data = historical_data.groupby("year")["price"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(historical_data["year"], historical_data["price"], marker="o", label="Historical Prices")
            ax.plot(year, predicted_price, "ro", label="Predicted Price")
            ax.axvline(x=year, color="r", linestyle="--", label="Forecast Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price (₹)")
            ax.set_title(f"Price Forecast for {crop} in {location}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No historical data available for the selected crop, location, and season.")

    # **LSTM Predictions Section**
    elif selected_page == "🤖 LSTM Predictions":
        st.subheader("🤖 LSTM Predictions")
        rainfall = st.slider("Rainfall (mm)", 0, 500, 200)
        temperature = st.slider("Temperature (°C)", 5, 50, 25)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        irrigation = st.selectbox("Select Irrigation Type", label_encoders["irrigation"].classes_)
        humidity = st.slider("Humidity (%)", 0, 100, 50)

        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        irrigation_encoded = label_encoders["irrigation"].transform([irrigation])[0]

        input_features = np.array([[rainfall, temperature, soil_type_encoded, irrigation_encoded, humidity]])
        input_features = input_features.reshape(1, 5, 1)

        if lstm_model:
            predicted_price = lstm_model.predict(input_features)[0][0]
            st.success(f"📈 Predicted Crop Price: ₹{predicted_price:.2f}")
        else:
            st.warning("⚠️ LSTM model not available. Please check the model file.")

    # **Optimal Crop Selection Section**
    elif selected_page == "🌾 Optimal Crop Selection":
        st.subheader("🌾 Optimal Crop Selection")
        rainfall = st.slider("Rainfall (mm)", 50, 500, 200)
        temperature = st.slider("Temperature (°C)", 10, 40, 25)
        soil_type = st.selectbox("Soil Type", label_encoders["soil_type"].classes_)
        season = st.selectbox("Select Season", label_encoders["season"].classes_)
        irrigation = st.selectbox("Select Irrigation Type", label_encoders["irrigation"].classes_)
        location = st.selectbox("Select Location", label_encoders["location"].classes_)
        humidity = st.slider("Humidity (%)", 0, 100, 50)

        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        season_encoded = label_encoders["season"].transform([season])[0]
        irrigation_encoded = label_encoders["irrigation"].transform([irrigation])[0]
        location_encoded = label_encoders["location"].transform([location])[0]

        features = [[rainfall, temperature, soil_type_encoded, humidity, season_encoded, irrigation_encoded, location_encoded]]
        probs = ml_models["crop"].predict_proba(features)[0]
        top_indices = np.argsort(probs)[::-1][:3]
        top_crops = label_encoders["crops"].inverse_transform(top_indices)
        top_probs = probs[top_indices]

        st.success("Top Recommended Crops:")
        for crop, prob in zip(top_crops, top_probs):
            st.write(f"{crop}: {prob:.2f}")

    # **XGBoost Predictions Section**
    elif selected_page == "📉 XGBoost Predictions":
        st.subheader("📉 Predict Crop Yield and Price Using XGBoost")
        rainfall = st.slider("Rainfall (mm)", 0, 500, 200)
        temperature = st.slider("Temperature (°C)", 5, 50, 25)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        irrigation = st.selectbox("Select Irrigation Type", label_encoders["irrigation"].classes_)
        year = st.slider("Select Year", 2000, 2030, 2025)
        location = st.selectbox("Select Location", label_encoders["location"].classes_)
        crops = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        season = st.selectbox("Select Season", label_encoders["season"].classes_)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        area = st.slider("Area (hectares)", 1, 100, 10)

        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        irrigation_encoded = label_encoders["irrigation"].transform([irrigation])[0]
        location_encoded = label_encoders["location"].transform([location])[0]
        crops_encoded = label_encoders["crops"].transform([crops])[0]
        season_encoded = label_encoders["season"].transform([season])[0]

        yield_features = np.array([[rainfall, temperature, soil_type_encoded, irrigation_encoded, humidity, area]])
        predicted_yield = ml_models["yield"].predict(yield_features)[0]

        price_features = np.array([[year, location_encoded, crops_encoded, predicted_yield, season_encoded]])
        predicted_price = ml_models["price"].predict(price_features)[0]

        revenue = predicted_yield * predicted_price
        st.success(f"🌾 Predicted Crop Yield: {predicted_yield:.2f} units")
        st.success(f"💰 Predicted Crop Price: ₹{predicted_price:.2f}")
        st.success(f"💵 Potential Revenue: ₹{revenue:.2f}")

    # **Irrigation Recommendation Section**
    elif selected_page == "💧 Irrigation Recommendation":
        st.subheader("💧 Predict Irrigation Type")
        selected_crop = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        rainfall = st.slider("Rainfall (mm)", 0, 500, 200)
        temperature = st.slider("Temperature (°C)", 5, 50, 25)

        selected_crop_encoded = label_encoders["crops"].transform([selected_crop])[0]
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]

        features = [[selected_crop_encoded, soil_type_encoded, rainfall, temperature]]
        probs = ml_models["irrigation"].predict_proba(features)[0]
        top_indices = np.argsort(probs)[::-1][:3]
        top_irrigation = label_encoders["irrigation"].inverse_transform(top_indices)
        top_probs = probs[top_indices]

        st.success("Top Recommended Irrigation Types:")
        for irr, prob in zip(top_irrigation, top_probs):
            st.write(f"{irr}: {prob:.2f}")

    # **Best Crop Prediction Section**
    elif selected_page == "🌱 Best Crop Prediction":
        st.subheader("🌱 Best Crop Prediction")
        st.write("Select a future season, soil type, and location to find the best crops based on predicted yield.")

        # User inputs
        season = st.selectbox("Select Season", label_encoders["season"].classes_)
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_)
        location = st.selectbox("Select Location", label_encoders["location"].classes_)

        # Encode selected values
        season_encoded = label_encoders["season"].transform([season])[0]
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
        location_encoded = label_encoders["location"].transform([location])[0]

        # Calculate average conditions based on historical data
        filtered_df = df[(df["season"] == season_encoded) & (df["location"] == location_encoded)]
        if filtered_df.empty:
            st.warning("No historical data available for the selected season and location.")
        else:
            avg_rainfall = filtered_df["rainfall"].mean()
            avg_temperature = filtered_df["temperature"].mean()
            avg_humidity = filtered_df["humidity"].mean()

            # Standardize area to 1 hectare for comparability
            area = 1

            # Get all possible crops
            possible_crops = label_encoders["crops"].classes_

            # Predict yield for each crop
            predictions = []
            for crop in possible_crops:
                crop_encoded = label_encoders["crops"].transform([crop])[0]

                # Determine most common irrigation method for this crop
                crop_df = df[df["crops"] == crop_encoded]
                if not crop_df.empty:
                    most_common_irrigation = crop_df["irrigation"].mode()[0]
                else:
                    most_common_irrigation = df["irrigation"].mode()[0]  # Fallback to overall most common

                # Create input features for prediction
                features = [[season_encoded, soil_type_encoded, location_encoded, crop_encoded,
                             avg_rainfall, avg_temperature, avg_humidity, most_common_irrigation, area]]

                # Predict yield using the new model
                predicted_yield = ml_models["yield_with_crop"].predict(features)[0]
                predictions.append((crop, predicted_yield))

            # Sort predictions by yield in descending order
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Display top 3 recommended crops
            st.success("Top Recommended Crops Based on Predicted Yield:")
            for crop, yield_pred in predictions[:3]:
                st.write(f"- **{crop}**: Predicted Yield = {yield_pred:.2f} units")

    # **Pest & Disease Scanner Section**
    elif selected_page == "📸 Pest & Disease Scanner":
        st.subheader("📸 Pest & Disease Scanner")
        st.write("Upload an image of your crop to detect pests or diseases.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Simulate image classification (replace with actual model in production)
            detections = ["Healthy", "Aphids", "Fungal Infection"]
            detection = np.random.choice(detections)
            recommendation = {
                "Healthy": "No action needed.",
                "Aphids": "Apply neem oil or insecticidal soap.",
                "Fungal Infection": "Use fungicide and improve air circulation."
            }[detection]

            st.success(f"Detection: {detection}")
            st.info(f"Recommendation: {recommendation}")

    # **Smart Irrigation Section**
    elif selected_page == "💧 Smart Irrigation":
        st.subheader("💧 Smart Irrigation")
        st.write("Get real-time irrigation recommendations based on simulated IoT data.")

        soil_moisture, temperature, humidity = get_iot_data()
        st.write(f"Current Conditions: Soil Moisture: {soil_moisture:.1f}%, Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%")

        recommendation = recommend_irrigation(soil_moisture, temperature, humidity)
        st.success(recommendation)

    # **Profit Simulator Section**
    elif selected_page == "🎲 Profit Simulator":
        # st.subheader("🎲 Profit Simulator")
        # st.write("Simulate potential profits based on different scenarios.")

        # crop = st.selectbox("Select Crop", label_encoders["crops"].classes_)
        # area = st.slider("Area (hectares)", 1, 100, 10)
        # investment = st.slider("Investment (₹)", 1000, 100000, 10000)

        # crop_encoded = label_encoders["crops"].transform([crop])[0]
        # location_encoded = label_encoders["location"].transform([label_encoders["location"].classes_[0]])[0]  # Default to first location
        # season_encoded = label_encoders["season"].transform([label_encoders["season"].classes_[0]])[0]  # Default to first season
        # avg_rainfall = df["rainfall"].mean()
        # avg_temperature = df["temperature"].mean()
        # avg_humidity = df["humidity"].mean()
        # most_common_irrigation = df["irrigation"].mode()[0]

        # # Simulate yield and price with uncertainty
        # num_simulations = 100
        # yields = np.random.normal(loc=ml_models["yield"].predict([[avg_rainfall, avg_temperature, soil_type_encoded, most_common_irrigation, avg_humidity, area]])[0], scale=10, size=num_simulations)
        # prices = np.random.normal(loc=ml_models["price"].predict([[2025, location_encoded, crop_encoded, yields.mean(), season_encoded]])[0], scale=5, size=num_simulations)
        # costs = investment * np.random.uniform(0.8, 1.2, size=num_simulations)

        # profits = (yields * prices) - costs

        # st.write(f"Average Profit: ₹{profits.mean():.2f}")
        # st.write(f"Profit Range: ₹{profits.min():.2f} to ₹{profits.max():.2f}")

        # fig, ax = plt.subplots()
        # ax.hist(profits, bins=20)
        # ax.set_title("Profit Distribution")
        # ax.set_xlabel("Profit (₹)")
        # ax.set_ylabel("Frequency")
        # st.pyplot(fig)
        st.subheader("🎲 Profit Simulator")
        st.write("Simulate potential profits based on different scenarios.")

    # User inputs
        crop = st.selectbox("Select Crop", label_encoders["crops"].classes_, key="profit_crop")
        location = st.selectbox("Select Location", label_encoders["location"].classes_, key="profit_location")
        season = st.selectbox("Select Season", label_encoders["season"].classes_, key="profit_season")
        soil_type = st.selectbox("Select Soil Type", label_encoders["soil_type"].classes_, key="profit_soil_type")
        area = st.slider("Area (hectares)", 1, 100, 10, key="profit_area")
        investment = st.slider("Investment (₹)", 1000, 100000, 10000, key="profit_investment")

    # Encode categorical variables
        crop_encoded = label_encoders["crops"].transform([crop])[0]
        location_encoded = label_encoders["location"].transform([location])[0]
        season_encoded = label_encoders["season"].transform([season])[0]
        soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]

    # Calculate averages based on location and season
        filtered_df = df[(df["location"] == location_encoded) & (df["season"] == season_encoded)]
        if not filtered_df.empty:
            avg_rainfall = filtered_df["rainfall"].mean()
            avg_temperature = filtered_df["temperature"].mean()
            avg_humidity = filtered_df["humidity"].mean()
        else:
            avg_rainfall = df["rainfall"].mean()
            avg_temperature = df["temperature"].mean()
            avg_humidity = df["humidity"].mean()

    # Get most common irrigation for the selected crop
        crop_df = df[df["crops"] == crop_encoded]
        if not crop_df.empty:
            most_common_irrigation = crop_df["irrigation"].mode()[0]
        else:
            most_common_irrigation = df["irrigation"].mode()[0]

    # Simulate yield and price with uncertainty
        num_simulations = 100
        yield_pred = ml_models["yield"].predict([[avg_rainfall, avg_temperature, soil_type_encoded, most_common_irrigation, avg_humidity, area]])[0]
        yields = np.random.normal(loc=yield_pred, scale=10, size=num_simulations)

        price_pred = ml_models["price"].predict([[2025, location_encoded, crop_encoded, yield_pred, season_encoded]])[0]
        prices = np.random.normal(loc=price_pred, scale=5, size=num_simulations)

    # Calculate profits
        costs = investment * np.random.uniform(0.8, 1.2, size=num_simulations)
        profits = (yields * prices) - costs

    # Display results
        st.write(f"Average Profit: ₹{profits.mean():.2f}")
        st.write(f"Profit Range: ₹{profits.min():.2f} to ₹{profits.max():.2f}")

    # Plot profit distribution
        fig, ax = plt.subplots()
        ax.hist(profits, bins=20)
        ax.set_title("Profit Distribution")
        ax.set_xlabel("Profit (₹)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # **Community Insights Section**
    elif selected_page == "🌐 Community Insights":
        st.subheader("🌐 Community Insights")
        location = st.selectbox("Select Your Location", label_encoders["location"].classes_)
        location_encoded = label_encoders["location"].transform([location])[0]

        insights = get_community_insights(location_encoded)
        if insights:
            top_crop, avg_yield = insights
            st.write(f"Top Crop in Your Area: {top_crop}")
            st.write(f"Average Yield: {avg_yield:.2f} units")
        else:
            st.warning("No community data available for this location.")

    # **Climate Planner Section**
    elif selected_page == "🌍 Climate Planner":
        st.subheader("🌍 Climate Planner")
        st.write("Get recommendations for adapting to future climate conditions.")

        current_rainfall, future_rainfall = get_climate_trend()
        st.write(f"Current Average Rainfall: {current_rainfall:.2f} mm")
        st.write(f"Predicted Future Rainfall: {future_rainfall:.2f} mm")

        recommendation = recommend_climate_resilient_crops(future_rainfall)
        st.success(recommendation)

if __name__ == "__main__":
    main()