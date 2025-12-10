import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os
import time

# Klasör Yolları
ARTIFACTS_DIR = 'artifacts'
OUTPUTS_DIR = 'outputs'

# ============================================================================
# 1. SAYFA YAPILANDIRMASI
# ============================================================================
st.set_page_config(
    page_title="Gazi Üni. - IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        height: 50px;
        font-weight: bold;
    }
    .status-normal { color: #28a745; font-weight: bold; font-size: 24px; }
    .status-attack { color: #dc3545; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. YARDIMCI FONKSİYONLAR
# ============================================================================

@st.cache_resource
def load_artifacts():
    try:
        # Modelleri artifacts klasöründen çek
        model_path = os.path.join(ARTIFACTS_DIR, 'best_intrusion_detection_model.pkl')
        scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        # Bilgileri outputs klasöründen çek
        best_dr_method = 'Original'
        best_model_name = 'Unknown'
        info_path = os.path.join(OUTPUTS_DIR, 'model_info.txt')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                lines = f.readlines()
                best_model_name = lines[0].strip()
                if len(lines) > 1:
                    best_dr_method = lines[1].strip()
        
        dr_model = None
        if best_dr_method != 'Original':
            from tensorflow.keras.models import load_model
            
            h5_path = os.path.join(ARTIFACTS_DIR, 'dim_reduction_model.h5')
            pkl_path = os.path.join(ARTIFACTS_DIR, 'dim_reduction_model.pkl')
            
            if best_dr_method == 'Autoencoder' and os.path.exists(h5_path):
                 dr_model = load_model(h5_path)
            elif os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    dr_model = pickle.load(f)
                    
        return model, scaler, dr_model, best_model_name, best_dr_method
    except FileNotFoundError:
        return None, None, None, None, None

@st.cache_data
def load_sample_data():
    # Modül src klasöründe olduğu için import şekli değişti
    from src import data_utils
    X_train_scaled, X_test_scaled, y_train, y_test, _, feature_names = data_utils.load_and_preprocess_data()
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

# ============================================================================
# 3. YAN MENÜ
# ============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("IDS Paneli")
    st.markdown("---")
    
    model, scaler, dr_model, model_name, dr_method = load_artifacts()
    
    if model:
        st.success(f"Model: **{model_name}**")
        st.info(f"Metod: **{dr_method}**")
    else:
        st.error(f"Dosyalar eksik! Lütfen '{ARTIFACTS_DIR}' klasörünü kontrol edin veya eğitimi çalıştırın.")
        st.stop()
        
    st.markdown("---")
    st.caption("Gazi Üniversitesi - Canberk Aykanat")

# ============================================================================
# 4. ANA EKRAN
# ============================================================================

st.title("🛡️ Network Intrusion Detection System")

tab1, tab2, tab3 = st.tabs([" Simülasyon", "📊 Grafikler", "🧠 Detaylar"])

# --- TAB 1 ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with st.spinner("Veri hazırlanıyor..."):
        X_train, X_test, y_train, y_test, feature_names = load_sample_data()

    with col1:
        st.markdown("### 📡 Kontrol")
        if st.button("Rastgele Paket Analiz Et"):
            import random
            random_idx = random.randint(0, len(X_test) - 1)
            st.session_state['input_vector'] = X_test[random_idx].reshape(1, -1)
            st.session_state['true_label'] = y_test.iloc[random_idx]
            st.session_state['analyzed'] = True
            
            with st.status("Analiz ediliyor...", expanded=True) as status:
                time.sleep(1)
                status.update(label="Tamamlandı!", state="complete", expanded=False)

    if 'analyzed' in st.session_state:
        input_vector = st.session_state['input_vector']
        true_label = st.session_state['true_label']
        
        # Boyut İndirgeme
        if dr_method != 'Original' and dr_model is not None:
            if dr_method == 'Autoencoder':
                processed_input = dr_model.predict(input_vector, verbose=0)
            elif hasattr(dr_model, 'transform'):
                processed_input = dr_model.transform(input_vector)
            else:
                processed_input = input_vector
        else:
            processed_input = input_vector

        # Tahmin
        prediction = model.predict(processed_input)[0]
        try:
            confidence = model.predict_proba(processed_input)[0][prediction]
        except:
            confidence = 1.0

        with col2:
            st.markdown("### 🔍 Sonuç")
            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.markdown('<div class="status-attack">🚨 SALDIRI TESPİT EDİLDİ</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-normal">✅ NORMAL TRAFİK</div>', unsafe_allow_html=True)
            with c2:
                st.metric("Güven Skoru", f"%{confidence*100:.1f}")
                st.caption(f"Gerçek Etiket: {'Saldırı' if true_label==1 else 'Normal'}")

        st.divider()
        feature_vals = input_vector[0][:20]
        df_chart = pd.DataFrame({'Özellik': [f'F{i}' for i in range(len(feature_vals))], 'Değer': feature_vals})
        fig = px.bar(df_chart, x='Özellik', y='Değer', title="Öznitelik Değerleri (İlk 20)", color='Değer')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2 ---
with tab2:
    st.header("Performans Görselleri")
    
    # Görselleri outputs klasöründen çek
    cm_path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    comp_path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        else:
            st.warning("Görsel henüz oluşturulmadı.")
    with col_g2:
        if os.path.exists(comp_path):
            st.image(comp_path, caption="Model Karşılaştırması", use_container_width=True)
        else:
            st.warning("Görsel henüz oluşturulmadı.")

# --- TAB 3 ---
with tab3:
    st.header("Model Bilgisi")
    st.info(f"Bu sistem **{model_name}** algoritmasını **{dr_method}** yöntemi ile kullanmaktadır.")