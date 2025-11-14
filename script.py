
# Create a comprehensive F1 ML prediction model implementation
# This will serve as the complete code that users can run

code_content = '''
# =====================================================
# F1 RACE PREDICTION MODEL USING FASTF1 API
# Complete Machine Learning Implementation
# =====================================================

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache for faster data loading
fastf1.Cache.enable_cache('f1_cache')

print("="*60)
print("F1 RACE PREDICTION MODEL - FASTF1 API")
print("="*60)


# =====================================================
# PART 1: DATA COLLECTION
# =====================================================

def collect_comprehensive_race_data(year, race_list, max_races=None):
    """
    Collects comprehensive race data including:
    - Qualifying results
    - Grid positions
    - Lap times and telemetry
    - Weather conditions
    - Tire strategies
    - Driver/team standings
    """
    all_data = []
    races_collected = 0
    
    for race_name in race_list:
        if max_races and races_collected >= max_races:
            break
            
        try:
            print(f"\\nCollecting data: {year} {race_name}")
            
            # Load race session
            race_session = fastf1.get_session(year, race_name, 'R')
            race_session.load()
            
            # Load qualifying session
            quali_session = fastf1.get_session(year, race_name, 'Q')
            quali_session.load()
            
            # Get race results
            race_results = race_session.results
            
            # Process each driver
            for idx, driver in race_results.iterrows():
                driver_number = driver['DriverNumber']
                
                # Get qualifying position
                quali_result = quali_session.results[
                    quali_session.results['DriverNumber'] == driver_number
                ]
                quali_pos = quali_result['Position'].values[0] if len(quali_result) > 0 else 20
                
                # Get driver laps
                driver_laps = race_session.laps.pick_driver(driver_number)
                
                # Calculate lap time statistics
                if len(driver_laps) > 0:
                    valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                    
                    if len(valid_laps) > 0:
                        fastest_lap = valid_laps['LapTime'].min().total_seconds()
                        avg_lap = valid_laps['LapTime'].mean().total_seconds()
                        lap_consistency = valid_laps['LapTime'].std().total_seconds()
                        total_laps = len(driver_laps)
                    else:
                        fastest_lap = avg_lap = lap_consistency = 0
                        total_laps = 0
                else:
                    fastest_lap = avg_lap = lap_consistency = total_laps = 0
                
                # Get weather data (average for the race)
                try:
                    weather_data = race_session.laps.get_weather_data()
                    if len(weather_data) > 0:
                        avg_track_temp = weather_data['TrackTemp'].mean()
                        avg_air_temp = weather_data['AirTemp'].mean()
                    else:
                        avg_track_temp = avg_air_temp = 0
                except:
                    avg_track_temp = avg_air_temp = 0
                
                # Count pit stops
                pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]
                num_pit_stops = len(pit_stops)
                
                # Create feature dictionary
                race_data = {
                    'Year': year,
                    'Race': race_name,
                    'Driver': driver['Abbreviation'],
                    'DriverNumber': driver_number,
                    'Team': driver['TeamName'],
                    'GridPosition': driver['GridPosition'],
                    'QualiPosition': quali_pos,
                    'FastestLapTime': fastest_lap,
                    'AverageLapTime': avg_lap,
                    'LapConsistency': lap_consistency,
                    'TotalLaps': total_laps,
                    'NumPitStops': num_pit_stops,
                    'TrackTemp': avg_track_temp,
                    'AirTemp': avg_air_temp,
                    'Position': driver['Position'],
                    'Points': driver['Points'],
                    'Status': driver['Status']
                }
                
                all_data.append(race_data)
            
            races_collected += 1
            print(f"✓ Collected {len(race_results)} drivers from {race_name}")
            
        except Exception as e:
            print(f"✗ Error with {race_name}: {str(e)}")
            continue
    
    return pd.DataFrame(all_data)


# =====================================================
# PART 2: DATA PREPROCESSING & FEATURE ENGINEERING
# =====================================================

def preprocess_and_engineer_features(df):
    """
    Clean data and create advanced features
    """
    print("\\n" + "="*60)
    print("DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)
    
    # Filter only finished races (remove DNF, DNS, etc.)
    df_clean = df[df['Status'].str.contains('Finished|Lap', na=False, case=False)].copy()
    print(f"Rows after filtering DNF/DNS: {len(df_clean)}")
    
    # Handle missing values
    df_clean['GridPosition'] = df_clean['GridPosition'].fillna(20)
    df_clean['QualiPosition'] = df_clean['QualiPosition'].fillna(20)
    df_clean['FastestLapTime'] = df_clean['FastestLapTime'].replace(0, 
                                    df_clean['FastestLapTime'].median())
    df_clean['AverageLapTime'] = df_clean['AverageLapTime'].replace(0, 
                                    df_clean['AverageLapTime'].median())
    df_clean['LapConsistency'] = df_clean['LapConsistency'].fillna(
                                    df_clean['LapConsistency'].median())
    
    # Encode categorical variables
    team_encoder = LabelEncoder()
    driver_encoder = LabelEncoder()
    
    df_clean['TeamEncoded'] = team_encoder.fit_transform(df_clean['Team'])
    df_clean['DriverEncoded'] = driver_encoder.fit_transform(df_clean['Driver'])
    
    # Advanced Feature Engineering
    
    # 1. Grid advantage (lower grid = better)
    df_clean['GridAdvantage'] = 21 - df_clean['GridPosition']
    
    # 2. Quali vs Grid difference (penalties indicator)
    df_clean['QualiGridDelta'] = df_clean['GridPosition'] - df_clean['QualiPosition']
    
    # 3. Relative pace (normalized lap time)
    race_fastest = df_clean.groupby('Race')['FastestLapTime'].transform('min')
    df_clean['RelativePace'] = (df_clean['FastestLapTime'] - race_fastest) / race_fastest
    
    # 4. Team performance indicator
    team_avg_position = df_clean.groupby('Team')['Position'].transform('mean')
    df_clean['TeamStrength'] = 21 - team_avg_position
    
    # 5. Driver historical performance
    driver_avg_position = df_clean.groupby('Driver')['Position'].transform('mean')
    df_clean['DriverSkill'] = 21 - driver_avg_position
    
    # 6. Pit stop efficiency
    df_clean['PitStopRate'] = df_clean['NumPitStops'] / (df_clean['TotalLaps'] + 1)
    
    # 7. Temperature normalized features
    df_clean['TempIndex'] = (df_clean['TrackTemp'] + df_clean['AirTemp']) / 2
    
    # 8. Consistency score (lower is better)
    df_clean['ConsistencyScore'] = df_clean['LapConsistency'] / (df_clean['AverageLapTime'] + 1)
    
    print(f"\\nFeatures created: {len(df_clean.columns)}")
    print(f"Final dataset size: {len(df_clean)} rows")
    
    return df_clean, team_encoder, driver_encoder


# =====================================================
# PART 3: MODEL TRAINING
# =====================================================

def train_prediction_models(df):
    """
    Train multiple models and compare performance
    """
    print("\\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Select features for training
    feature_columns = [
        'GridPosition', 'QualiPosition', 'FastestLapTime', 'AverageLapTime',
        'LapConsistency', 'TotalLaps', 'NumPitStops', 'TrackTemp', 'AirTemp',
        'TeamEncoded', 'DriverEncoded', 'GridAdvantage', 'QualiGridDelta',
        'RelativePace', 'TeamStrength', 'DriverSkill', 'PitStopRate',
        'TempIndex', 'ConsistencyScore'
    ]
    
    X = df[feature_columns]
    y = df['Position']
    
    print(f"\\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Model 1: Random Forest
    print("\\n--- Training Random Forest ---")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # Model 2: Gradient Boosting
    print("--- Training Gradient Boosting ---")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    gb_train_pred = gb_model.predict(X_train)
    gb_test_pred = gb_model.predict(X_test)
    
    # Evaluate models
    print("\\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    
    # Random Forest metrics
    print("\\n--- Random Forest Regressor ---")
    print(f"Train MAE: {mean_absolute_error(y_train, rf_train_pred):.3f} positions")
    print(f"Test MAE: {mean_absolute_error(y_test, rf_test_pred):.3f} positions")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, rf_test_pred)):.3f}")
    print(f"Test R² Score: {r2_score(y_test, rf_test_pred):.3f}")
    
    # Gradient Boosting metrics
    print("\\n--- Gradient Boosting Regressor ---")
    print(f"Train MAE: {mean_absolute_error(y_train, gb_train_pred):.3f} positions")
    print(f"Test MAE: {mean_absolute_error(y_test, gb_test_pred):.3f} positions")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, gb_test_pred)):.3f}")
    print(f"Test R² Score: {r2_score(y_test, gb_test_pred):.3f}")
    
    # Feature importance
    print("\\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Select best model
    rf_mae = mean_absolute_error(y_test, rf_test_pred)
    gb_mae = mean_absolute_error(y_test, gb_test_pred)
    
    best_model = rf_model if rf_mae < gb_mae else gb_model
    best_model_name = "Random Forest" if rf_mae < gb_mae else "Gradient Boosting"
    
    print(f"\\n✓ Best Model: {best_model_name}")
    
    return best_model, feature_columns, X_test, y_test


# =====================================================
# PART 4: PREDICTION FUNCTION
# =====================================================

def predict_race_results(model, feature_columns, new_data):
    """
    Predict race finishing positions for new race data
    """
    predictions = model.predict(new_data[feature_columns])
    predicted_positions = np.clip(np.round(predictions), 1, 20).astype(int)
    
    return predicted_positions


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    # Define 2024 season races
    races_2024 = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
        'Miami', 'Imola', 'Monaco', 'Canada', 'Spain',
        'Austria', 'Britain', 'Hungary', 'Belgium'
    ]
    
    # Step 1: Collect data (using first 5 races for demo)
    print("\\nStep 1: Collecting race data from FastF1...")
    df_raw = collect_comprehensive_race_data(2024, races_2024, max_races=5)
    
    if len(df_raw) > 0:
        # Step 2: Preprocess and engineer features
        print("\\nStep 2: Preprocessing data...")
        df_processed, team_enc, driver_enc = preprocess_and_engineer_features(df_raw)
        
        # Step 3: Train models
        print("\\nStep 3: Training prediction models...")
        model, features, X_test, y_test = train_prediction_models(df_processed)
        
        # Step 4: Sample prediction
        print("\\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        
        sample_predictions = predict_race_results(model, features, X_test[:10])
        actual_positions = y_test.iloc[:10].values
        
        comparison_df = pd.DataFrame({
            'Actual Position': actual_positions,
            'Predicted Position': sample_predictions,
            'Error': np.abs(actual_positions - sample_predictions)
        })
        
        print(comparison_df.to_string(index=False))
        
        print("\\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE!")
        print("="*60)
        print("\\nNext Steps:")
        print("1. Collect more historical data (2018-2024) for better accuracy")
        print("2. Add driver/team championship standings as features")
        print("3. Incorporate practice session data")
        print("4. Implement ensemble methods (stacking, voting)")
        print("5. Use the model to predict upcoming race results")
        
    else:
        print("\\n✗ No data collected. Check your internet connection.")
'''

# Save the code to a file
with open('f1_ml_model_complete.py', 'w', encoding='utf-8') as f:
    f.write(code_content)

print("✓ Complete F1 ML model code saved to: f1_ml_model_complete.py")
print(f"File size: {len(code_content)} characters")
print("\nThe file includes:")
print("1. Data collection from FastF1 API")
print("2. Comprehensive feature engineering")
print("3. Random Forest and Gradient Boosting models")
print("4. Model evaluation and comparison")
print("5. Prediction functionality")
