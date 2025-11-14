F1 Race Podium Predictor

Uses FastF1 telemetry + historical race data to train a machine-learning model that predicts podium probability for each driver in an upcoming Formula 1 race.

🚀 Features

Downloads race + qualifying data with FastF1

Automatically caches all sessions for faster repeated runs

Builds simple but effective features:

Qualifying position

Driver recent form

Team recent form

Season-to-date averages

Trains an XGBoost classifier to predict podium (Top-3)

Produces a sorted probability table for any upcoming race

📦 Installation
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


Or install manually:

pip install fastf1 pandas numpy scikit-learn xgboost tqdm

🗂 Project Structure
.
├── f1_podium_predictor.py   # main script (data, training, prediction)
├── requirements.txt         # dependencies
└── README.md

⚙️ Usage
1. Prepare FastF1 cache directory

FastF1 requires a physical folder for caching sessions.

In the script, this is handled automatically:

os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

2. Run the script
python f1_podium_predictor.py


It will:

Build a historical dataset

Train an XGBoost model

Print accuracy + AUC

Tell you how to get predictions for an upcoming race

🎯 Predicting an Upcoming Race

Inside the script:

predict_for_upcoming(model, feats, year=2024, round=7)


Example:

preds = predict_for_upcoming(model, feats, 2024, 7)
print(preds)


Output will look like:

   driver   team        podium_prob
0   VER     Red Bull      0.81
1   NOR     McLaren       0.32
2   LEC     Ferrari       0.29
...

📈 Model

Type: XGBoost binary classifier

Objective: Predict probability of finishing Top 3

Training data: 2019–2024 FastF1 race weekend results

Features:

Qualifying position

Driver recent finishes

Team recent finishes

Season average performance

This is a baseline model. Real predictive power comes once telemetry, practice pace, weather, and tire data are added.

📥 Requirements

requirements.txt example:

fastf1
pandas
numpy
scikit-learn
xgboost
tqdm

⚠️ Notes & Limitations

FastF1 may occasionally fail to load certain older sessions

Podium prediction is inherently noisy — treat outputs as probabilities, not truth

Telemetry analysis can be added later for better accuracy

🏎️ Future Improvements

Add practice session pace deltas

Integrate track-specific performance history

Use race stint data (tyres, lap-by-lap pace)

Add a UI dashboard for prediction visualization
