import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load('cricket_model1.pkl')
scaler = joblib.load('scaler.pkl1')

teams = ['Select', 'Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Titans',
         'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Delhi Capitals', 'Punjab Kings', 'Chennai Super Kings',
         'Rajasthan Royals']

cities = ['Select', 'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

le_team = LabelEncoder()
le_city = LabelEncoder()

le_team.fit(teams)
le_city.fit(cities)

st.title("IPL Winning Predictor")

batting_team = st.selectbox('Batting Team', teams)
bowling_team = st.selectbox('Bowling Team', teams)
city = st.selectbox('City', cities)
target_runs = st.number_input('Target Runs', min_value=0)
runs_left = st.number_input('Runs Left', min_value=0)
balls_left = st.number_input('Balls Left', min_value=0)
wickets_left = st.number_input('Wickets Left', min_value=0)

input_data = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'Runs_left': [runs_left],
    'Balls_left': [balls_left],
    'Wickets_left': [wickets_left],
    'total_runs_x': [target_runs],
    'CRR': [(target_runs - runs_left) * 6 / (120 - balls_left)],
    'RRR': [runs_left * 6 / balls_left] if balls_left > 0 else [0],
    'Result': [0]  
})


input_data['batting_team'] = le_team.transform(input_data['batting_team'])
input_data['bowling_team'] = le_team.transform(input_data['bowling_team'])
input_data['city'] = le_city.transform(input_data['city'])


columns_used = ['batting_team', 'city', 'bowling_team', 'Runs_left', 'Balls_left', 'Wickets_left', 'total_runs_x', 'CRR', 'RRR']


input_data = input_data[columns_used]


scaled_input_data = scaler.transform(input_data)

if st.button('Predict'):
    probability = model.predict_proba(scaled_input_data)[:, 1][0]
    
    if balls_left == 0 or wickets_left == 0:
        probability = 0
    else:
        probability = model.predict_proba(scaled_input_data)[:, 1][0]

    st.write(f'The probability of the {batting_team} winning is {probability:.2f}')
