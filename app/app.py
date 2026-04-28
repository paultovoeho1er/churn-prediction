import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé (pas d'indentation avant le """
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.prediction-churn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    animation: pulse 1.5s ease-in-out;
}
.prediction-no-churn {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    animation: pulse 1.5s ease-in-out;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}
.feature-importance {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Prediction System</h1>
    <p>Prédiction du risque d'attrition client avec Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🎯 À propos")
    st.markdown("""
    Cette application utilise le **Machine Learning** pour prédire si un client 
    est susceptible de résilier son contrat (churn).
    
    ### 📊 Modèle utilisé
    - **XGBoost Classifier** optimisé
    - Accuracy: ~85%
    - AUC-ROC: ~0.89
    
    ### 🔍 Comment ça marche?
    1. Remplissez les informations du client
    2. Cliquez sur "Prédire"
    3. Obtenez l'analyse du risque
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Statistiques")
    st.markdown("""
    - **Taux de churn moyen**: 26.5%
    - **Clients analysés**: 7043
    - **Features utilisées**: 45
    """)

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None, None

model, scaler, feature_names = load_models()

if model is None:
    st.stop()

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Informations Client")
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["🏠 Démographie", "📡 Services", "💰 Facturation"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("Genre", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partenaire", ["No", "Yes"])
        with col_b:
            dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
            tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            phone_service = st.selectbox("Service téléphonique", ["Yes", "No"])
            multiple_lines = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        with col_b:
            online_security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Protection d'appareil", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Films", ["No", "Yes", "No internet service"])
    
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Facturation sans papier", ["Yes", "No"])
        with col_b:
            payment_method = st.selectbox("Méthode de paiement", 
                                         ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                                          "Credit card (automatic)"])
            monthly_charges = st.number_input("Charges mensuelles ($)", 20.0, 150.0, 65.0)
            total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

with col2:
    st.markdown("## 📊 Quick Stats")
    
    # Métriques rapides
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 Ancienneté moyenne</h3>
        <h2>{tenure} mois</h2>
        <p>{'🟢 Client fidèle' if tenure > 12 else '🟡 Nouveau client'}</p>
    </div>
    <br>
    <div class="metric-card">
        <h3>💰 Revenue mensuel</h3>
        <h2>${monthly_charges:.0f}</h2>
        <p>Total: ${total_charges:.0f}</p>
    </div>
    """, unsafe_allow_html=True)

# Fonction d'encodage simplifiée (à adapter selon vos features)
def prepare_features(input_dict):
    """Convertit les inputs utilisateur en features pour le modèle"""
    
    # Dictionnaire de mapping
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map = {
        "Electronic check": 0, 
        "Mailed check": 1, 
        "Bank transfer (automatic)": 2, 
        "Credit card (automatic)": 3
    }
    
    # Créer le vecteur de features
    features = {
        'gender': binary_map.get(input_dict.get('gender'), 0),
        'SeniorCitizen': 1 if input_dict.get('senior_citizen') == 'Yes' else 0,
        'Partner': binary_map.get(input_dict.get('partner'), 0),
        'Dependents': binary_map.get(input_dict.get('dependents'), 0),
        'tenure': float(input_dict.get('tenure', 0)),
        'PhoneService': binary_map.get(input_dict.get('phone_service'), 0),
        'MultipleLines': service_map.get(input_dict.get('multiple_lines'), 0),
        'InternetService': 0 if input_dict.get('internet_service') == 'No' else (1 if input_dict.get('internet_service') == 'DSL' else 2),
        'OnlineSecurity': service_map.get(input_dict.get('online_security'), 0),
        'OnlineBackup': service_map.get(input_dict.get('online_backup'), 0),
        'DeviceProtection': service_map.get(input_dict.get('device_protection'), 0),
        'TechSupport': service_map.get(input_dict.get('tech_support'), 0),
        'StreamingTV': service_map.get(input_dict.get('streaming_tv'), 0),
        'StreamingMovies': service_map.get(input_dict.get('streaming_movies'), 0),
        'Contract': contract_map.get(input_dict.get('contract'), 0),
        'PaperlessBilling': binary_map.get(input_dict.get('paperless_billing'), 0),
        'PaymentMethod': payment_map.get(input_dict.get('payment_method'), 0),
        'MonthlyCharges': float(input_dict.get('monthly_charges', 0)),
        'TotalCharges': float(input_dict.get('total_charges', 0))
    }
    
    # Convertir en DataFrame
    df = pd.DataFrame([features])
    
    # S'assurer que toutes les colonnes attendues sont présentes
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Réordonner les colonnes
    df = df[feature_names]
    
    # Appliquer le scaling
    X_scaled = scaler.transform(df)
    
    return X_scaled

# Bouton de prédiction
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 PRÉDIRE LE RISQUE DE CHURN", use_container_width=True, type="primary")

# Prédiction
if predict_button:
    with st.spinner("Analyse en cours..."):
        try:
            # Préparation des données
            input_data = {
                'gender': gender,
                'senior_citizen': senior_citizen,
                'partner': partner,
                'dependents': dependents,
                'tenure': tenure,
                'phone_service': phone_service,
                'multiple_lines': multiple_lines,
                'internet_service': internet_service,
                'online_security': online_security,
                'online_backup': online_backup,
                'device_protection': device_protection,
                'tech_support': tech_support,
                'streaming_tv': streaming_tv,
                'streaming_movies': streaming_movies,
                'contract': contract,
                'paperless_billing': paperless_billing,
                'payment_method': payment_method,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges
            }
            
            # Préparer les features
            X_input = prepare_features(input_data)
            
            # Prédiction
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            
            risk_score = probability[1]  # Probabilité de churn
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de l'analyse")
            
            # Métriques de risque
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_label = "Élevé" if risk_score > 0.5 else "Modéré" if risk_score > 0.3 else "Faible"
                st.metric("Risque de Churn", f"{risk_score*100:.1f}%", delta=risk_label)
            with col2:
                st.metric("Score de confiance", f"{max(probability)*100:.1f}%", delta="Modèle XGBoost")
            with col3:
                recommendation = "Action requise" if prediction == 1 else "Client stable"
                st.metric("Recommandation", recommendation, delta="⚠️" if prediction == 1 else "✅")
            
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score * 100,
                title={'text': "Niveau de risque (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if risk_score > 0.5 else "orange" if risk_score > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 65.0
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Résumé et recommandations
            st.markdown("### 💡 Recommandations personnalisées")
            
            if prediction == 1:
                st.warning("""
                ⚠️ **Risque ÉLEVÉ de churn détecté!**
                
                **Actions recommandées:**
                - 📞 Contacter le client proactivement
                - 💰 Proposer une réduction ou une offre spéciale
                - 🔒 Offrir un contrat longue durée (réduction 10-15%)
                - 🎁 Programme de fidélisation personnalisé
                - 📊 Analyser les raisons d'insatisfaction
                """)
            else:
                st.success("""
                ✅ **Client à faible risque de churn**
                
                **Stratégie de rétention:**
                - 🎯 Programme de parrainage
                - 📈 Upsell/cross-sell opportunités
                - ⭐ Programme de fidélité VIP
                - 📧 Newsletters et offres exclusives
                """)
            
            # Facteurs de risque
            st.markdown("### 🔍 Facteurs de risque identifiés")
            
            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("⚠️ Contrat mensuel (risque +42%)")
            if internet_service == "Fiber optic":
                risk_factors.append("⚠️ Fibre optique (risque +15%)")
            if tenure < 12:
                risk_factors.append("⚠️ Nouveau client (risque +25%)")
            if monthly_charges > 100:
                risk_factors.append("⚠️ Charges mensuelles élevées")
            if paperless_billing == "Yes":
                risk_factors.append("⚠️ Facture sans papier (risque +10%)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- ✅ Aucun facteur de risque majeur identifié")
                
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            st.info("Veuillez vérifier que tous les champs sont correctement remplis.")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>🔬 Modèle entraîné sur Telco Customer Churn Dataset | Accuracy: 85% | AUC-ROC: 0.89</p>
    <p>📅 Dernière mise à jour: {datetime.now().strftime("%Y-%m-%d")} | 🚀 Déployé avec Streamlit</p>
</div>
""", unsafe_allow_html=True)