import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # Load processed training data
    df = pd.read_csv("data/processed/train_data_processed.csv")
    features = ['PTS', 'AST', 'TRB', 'FG%', 'MP']
    
    X = df[features]
    y = df['MVP']
    
    # Initialize and train the best performing model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the trained model to disk
    joblib.dump(model, 'models/nba_mvp_model.joblib')
    print(" Success: Best model saved as 'models/nba_mvp_model.joblib'.")

    # Simple validation (Test on the most recent season in training data)
    last_year = df['Season'].max()
    test_data = df[df['Season'] == last_year]
    probs = model.predict_proba(test_data[features])[:, 1]
    winner = test_data.iloc[probs.argmax()]['Player']
    print(f" Validation: {last_year} season predicted winner -> {winner}")

if __name__ == "__main__":
    train_and_save_model()