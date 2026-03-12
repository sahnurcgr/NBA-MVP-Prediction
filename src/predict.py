import pandas as pd
import joblib
import sys

# Configure stdout for UTF-8 to handle special characters/emojis correctly
sys.stdout.reconfigure(encoding='utf-8')

def predict_2025(file_path):
    try:
        # Load saved tools and model
        model = joblib.load('models/nba_mvp_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        imputer = joblib.load('models/imputer.joblib')
        
        # Load 2025 data
        df_2025_all = pd.read_csv(file_path)
        df_2025 = df_2025_all[df_2025_all['Season'] == 2025].copy()
        
        features = ['PTS', 'AST', 'TRB', 'FG%', 'MP']
        X_raw = df_2025[features]
        
        # Preprocessing using saved tools
        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)
        
        # Keep feature names to avoid UserWarnings
        X_final = pd.DataFrame(X_scaled, columns=features)
        
        # Predict probabilities
        probs = model.predict_proba(X_final)[:, 1]
        df_2025['MVP_Prob'] = probs
        
        # Get Top 5 Candidates
        top5 = df_2025.sort_values(by='MVP_Prob', ascending=False).head(5)
        
        print("\n🎯 2025 NBA MVP Prediction List")
        print("-" * 45)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"{i}. {row['Player']} | Probability: %{row['MVP_Prob']*100:.2f}")
            
    except FileNotFoundError:
        print(" Error: Files not found. Please run data_prep.py and train.py first.")

if __name__ == "__main__":
    predict_2025("data/raw/+2025.csv")