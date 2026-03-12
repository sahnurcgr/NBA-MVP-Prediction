import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def prepare_data(file_path, output_name):
    # Ensure the directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df = pd.read_csv(file_path)
    features = ['PTS', 'AST', 'TRB', 'FG%', 'MP']
    
    # Missing values and Scaling
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    # Process data
    scaled_values = imputer.fit_transform(df[features])
    scaled_values = scaler.fit_transform(scaled_values)
    
    # Create new DataFrame
    scaled_df = pd.DataFrame(scaled_values, columns=features)
    scaled_df['Player'] = df['Player']
    scaled_df['Season'] = df['Season']
    if 'MVP' in df.columns:
        scaled_df['MVP'] = df['MVP']
    
    # Save files
    scaled_df.to_csv(f"data/processed/{output_name}.csv", index=False)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(imputer, 'models/imputer.joblib')
    print(f" Success: {output_name} prepared and utility tools saved.")

if __name__ == "__main__":
    prepare_data("data/raw/last20yearsDatas.csv", "train_data_processed")