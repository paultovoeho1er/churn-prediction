# utils/encoder.py
encoder_code = """
import pandas as pd
import numpy as np

class ChurnDataEncoder:
    """Encodage des données pour la prédiction de churn"""
    
    def __init__(self, scaler, feature_names):
        self.scaler = scaler
        self.feature_names = feature_names
        
    def encode(self, input_data):
        # Mapping des valeurs
        self.binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        self.service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        
        # Construction du feature vector
        features = {}
        
        # Démographie
        features['gender'] = self.binary_map.get(input_data.get('gender'), 0)
        features['SeniorCitizen'] = 1 if input_data.get('senior_citizen') == 'Yes' else 0
        features['Partner'] = self.binary_map.get(input_data.get('partner'), 0)
        features['Dependents'] = self.binary_map.get(input_data.get('dependents'), 0)
        
        # Services
        features['PhoneService'] = self.binary_map.get(input_data.get('phone_service'), 0)
        features['MultipleLines'] = self.service_map.get(input_data.get('multiple_lines'), 0)
        
        # Internet services
        features['OnlineSecurity'] = self.service_map.get(input_data.get('online_security'), 0)
        features['OnlineBackup'] = self.service_map.get(input_data.get('online_backup'), 0)
        features['DeviceProtection'] = self.service_map.get(input_data.get('device_protection'), 0)
        features['TechSupport'] = self.service_map.get(input_data.get('tech_support'), 0)
        features['StreamingTV'] = self.service_map.get(input_data.get('streaming_tv'), 0)
        features['StreamingMovies'] = self.service_map.get(input_data.get('streaming_movies'), 0)
        
        # Contrat et facturation
        features['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(input_data.get('contract'), 0)
        features['PaperlessBilling'] = self.binary_map.get(input_data.get('paperless_billing'), 0)
        features['PaymentMethod'] = {
            "Electronic check": 0, "Mailed check": 1, 
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
        }.get(input_data.get('payment_method'), 0)
        
        # Charges
        features['tenure'] = float(input_data.get('tenure', 0))
        features['MonthlyCharges'] = float(input_data.get('monthly_charges', 0))
        features['TotalCharges'] = float(input_data.get('total_charges', 0))
        
        # One-hot encoding pour certaines features (à compléter selon votre modèle)
        # Ici vous devez reproduire exactement l'encodage de l'entraînement
        
        # Convertir en DataFrame
        df = pd.DataFrame([features])
        
        # S'assurer que toutes les colonnes sont présentes
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Réordonner les colonnes
        df = df[self.feature_names]
        
        # Scaling
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
"""

with open('app/utils/encoder.py', 'w', encoding='utf-8') as f:
    f.write(encoder_code)

print("✅ utils/encoder.py créé")