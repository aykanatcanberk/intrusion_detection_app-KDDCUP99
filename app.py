import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime
import random

# Klasör Yolları
ARTIFACTS_DIR = 'artifacts'
OUTPUTS_DIR = 'outputs'

# ============================================================================
# 1. SAYFA YAPILANDIRMASI
# ============================================================================
st.set_page_config(
    page_title="Advanced IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 50px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .status-normal { 
        color: #10b981; 
        font-weight: bold; 
        font-size: 28px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .status-attack { 
        color: #ef4444; 
        font-weight: bold; 
        font-size: 28px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. YARDIMCI FONKSİYONLAR
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Model ve artifactleri yükler"""
    try:
        model_path = os.path.join(ARTIFACTS_DIR, 'best_intrusion_detection_model.pkl')
        scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
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
    except FileNotFoundError as e:
        st.error(f"Dosya bulunamadı: {e}")
        return None, None, None, None, None

@st.cache_data
def load_sample_data():
    """Test verisi yükler"""
    from src import data_utils
    X_train_scaled, X_test_scaled, y_train, y_test, _, feature_names = \
        data_utils.load_and_preprocess_data()
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

@st.cache_data
def load_results():
    """Tüm sonuçları yükler"""
    results_path = os.path.join(OUTPUTS_DIR, 'all_model_results.csv')
    cv_path = os.path.join(OUTPUTS_DIR, 'cross_validation_results.csv')
    pipeline_path = os.path.join(OUTPUTS_DIR, 'preprocessing_pipeline.csv')
    
    results_df = None
    cv_df = None
    pipeline_df = None
    
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path)
    if os.path.exists(pipeline_path):
        pipeline_df = pd.read_csv(pipeline_path)
    
    return results_df, cv_df, pipeline_df

def predict_sample(model, dr_model, dr_method, input_vector):
    """Tek bir sample için tahmin yapar"""
    if dr_method != 'Original' and dr_model is not None:
        if dr_method == 'Autoencoder':
            processed_input = dr_model.predict(input_vector, verbose=0)
        elif hasattr(dr_model, 'transform'):
            processed_input = dr_model.transform(input_vector)
        else:
            processed_input = input_vector
    else:
        processed_input = input_vector

    prediction = model.predict(processed_input)[0]
    
    try:
        confidence = model.predict_proba(processed_input)[0][prediction]
    except:
        confidence = 1.0
    
    return prediction, confidence

# ============================================================================
# 3. YAN MENÜ
# ============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("🛡️ IDS Control Panel")
    st.markdown("---")
    
    model, scaler, dr_model, model_name, dr_method = load_artifacts()
    
    if model:
        st.success(f"✅ Model: **{model_name}**")
        st.info(f"📊 Method: **{dr_method}**")
        st.metric("Status", "Active", delta="Running")
    else:
        st.error("⚠️ Model bulunamadı!")
        st.stop()
        
    st.markdown("---")
    
    # Sistem İstatistikleri
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
        st.session_state.total_attacks = 0
        st.session_state.total_normal = 0
    
    st.markdown("### 📈 Oturum İstatistikleri")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Analiz", st.session_state.total_predictions)
    with col2:
        st.metric("Tespit Edilen", st.session_state.total_attacks)
    
    st.markdown("---")
    st.caption("🎓 Gazi Üniversitesi")
    st.caption("👨‍💻 Canberk Aykanat")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================================================
# 4. ANA EKRAN
# ============================================================================

st.title("🛡️ Advanced Network Intrusion Detection System")
st.markdown("### Gerçek Zamanlı Ağ Güvenliği İzleme ve Analiz Platformu")

# Ana Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Ana Dashboard", 
    "📊 Model Performansı", 
    "🔬 Detaylı Analiz",
    "📈 Veri İşleme",
    "⚙️ Toplu İşlem"
])

# ============================================================================
# TAB 1: ANA DASHBOARD
# ============================================================================
with tab1:
    # Veri yükleme
    with st.spinner("Veri yükleniyor..."):
        X_train, X_test, y_train, y_test, feature_names = load_sample_data()
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### 🎮 Kontrol Paneli")
        
        analysis_type = st.radio(
            "Analiz Tipi Seçin:",
            ["🎲 Rastgele Paket", "📝 Manuel Seçim"],
            horizontal=True
        )
        
        if analysis_type == "🎲 Rastgele Paket":
            if st.button("🔍 Analiz Başlat", use_container_width=True):
                random_idx = random.randint(0, len(X_test) - 1)
                st.session_state['input_vector'] = X_test[random_idx].reshape(1, -1)
                st.session_state['true_label'] = y_test.iloc[random_idx]
                st.session_state['sample_idx'] = random_idx
                st.session_state['analyzed'] = True
                st.session_state['total_predictions'] += 1
                
                with st.status("Analiz ediliyor...", expanded=True) as status:
                    st.write("🔄 Veri işleniyor...")
                    time.sleep(0.5)
                    st.write("🧠 Model tahmin yapıyor...")
                    time.sleep(0.5)
                    st.write("✅ Analiz tamamlandı!")
                    status.update(label="Tamamlandı!", state="complete", expanded=False)
                
                st.rerun()
        else:
            sample_idx = st.number_input(
                "Sample Index (0-{}):".format(len(X_test)-1),
                min_value=0, max_value=len(X_test)-1, value=0
            )
            if st.button("📊 Bu Sample'ı Analiz Et", use_container_width=True):
                st.session_state['input_vector'] = X_test[sample_idx].reshape(1, -1)
                st.session_state['true_label'] = y_test.iloc[sample_idx]
                st.session_state['sample_idx'] = sample_idx
                st.session_state['analyzed'] = True
                st.session_state['total_predictions'] += 1
                st.rerun()
    
    with col2:
        st.markdown("### 📡 Canlı Monitöring")
        
        if 'analyzed' in st.session_state:
            input_vector = st.session_state['input_vector']
            true_label = st.session_state['true_label']
            sample_idx = st.session_state['sample_idx']
            
            # Tahmin
            prediction, confidence = predict_sample(model, dr_model, dr_method, input_vector)
            
            if prediction == 1:
                st.session_state.total_attacks += 1
            else:
                st.session_state.total_normal += 1
            
            # Sonuç Kartları
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sample ID", f"#{sample_idx}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown('<div class="status-attack">🚨 SALDIRI!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-normal">✅ GÜVENLİ</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Güven Skoru", f"%{confidence*100:.1f}")
                st.progress(confidence)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Detaylar
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### 🎯 Tahmin Detayları")
                st.write(f"**Gerçek Etiket:** {'🔴 Saldırı' if true_label==1 else '🟢 Normal'}")
                st.write(f"**Tahmin:** {'🔴 Saldırı' if prediction==1 else '🟢 Normal'}")
                st.write(f"**Doğruluk:** {'✅ Doğru' if prediction==true_label else '❌ Yanlış'}")
            
            with col_d2:
                st.markdown("#### ⏱️ İşlem Bilgileri")
                st.write(f"**Analiz Zamanı:** {datetime.now().strftime('%H:%M:%S')}")
                st.write(f"**Model:** {model_name}")
                st.write(f"**Metod:** {dr_method}")
            
            st.divider()
            
            # Feature Visualization
            st.markdown("#### 📊 Öznitelik Analizi")
            
            feature_count = st.slider("Kaç özellik gösterilsin?", 10, 40, 20)
            feature_vals = input_vector[0][:feature_count]
            
            df_chart = pd.DataFrame({
                'Özellik': [f'F{i}' for i in range(len(feature_vals))],
                'Değer': feature_vals
            })
            
            fig = px.bar(df_chart, x='Özellik', y='Değer',
                        title=f"Öznitelik Değerleri (İlk {feature_count})",
                        color='Değer',
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Lütfen sol panelden bir analiz başlatın")
            
            # Placeholder animasyon
            st.markdown("### 🔄 Sistem Hazır")
            st.image("https://cdn-icons-png.flaticon.com/512/6243/6243789.png", width=200)

# ============================================================================
# TAB 2: MODEL PERFORMANSI
# ============================================================================
with tab2:
    st.header("📊 Model Performans Metrikleri")
    
    results_df, cv_df, pipeline_df = load_results()
    
    if results_df is not None:
        # En iyi model bilgisi
        best_model = results_df.loc[results_df['F1-Score'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("En İyi Model", best_model['Model'])
        with col2:
            st.metric("F1-Score", f"{best_model['F1-Score']:.4f}")
        with col3:
            st.metric("Accuracy", f"{best_model['Test Accuracy']:.4f}")
        with col4:
            st.metric("Eğitim Süresi", f"{best_model['Training Time']:.2f}s")
        
        st.divider()
        
        # Sonuç Tablosu
        st.markdown("### 📋 Tüm Model Sonuçları")
        
        # Filtreleme
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_models = st.multiselect(
                "Model Seç:",
                options=results_df['Model'].unique(),
                default=results_df['Model'].unique()
            )
        with col_f2:
            selected_dr = st.multiselect(
                "Boyut İndirgeme Seç:",
                options=results_df['Dimension Reduction'].unique(),
                default=results_df['Dimension Reduction'].unique()
            )
        
        filtered_df = results_df[
            (results_df['Model'].isin(selected_models)) &
            (results_df['Dimension Reduction'].isin(selected_dr))
        ]
        
        # Renklendirme ile tablo
        st.dataframe(
            filtered_df.style.background_gradient(subset=['F1-Score', 'Test Accuracy'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        st.divider()
        
        # Görselleştirmeler
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("### 🎯 F1-Score Karşılaştırması")
            fig = px.bar(filtered_df, x='Model', y='F1-Score',
                        color='Dimension Reduction',
                        barmode='group',
                        title="Model F1-Score Performansları")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            st.markdown("### ⏱️ Eğitim Süresi Analizi")
            fig = px.scatter(filtered_df, x='Training Time', y='F1-Score',
                           size='Test Accuracy', color='Model',
                           hover_data=['Dimension Reduction'],
                           title="Performans vs Süre")
            st.plotly_chart(fig, use_container_width=True)
    
    # Görsel Galeri
    st.divider()
    st.markdown("### 🖼️ Detaylı Performans Grafikleri")
    
    image_files = {
        'Confusion Matrix': 'confusion_matrix.png',
        'ROC Curves': 'roc_curves.png',
        'Precision-Recall': 'precision_recall_curves.png',
        'Learning Curve': 'learning_curve.png',
        'Cross-Validation': 'cross_validation_plot.png',
        'Model Comparison': 'model_comparison.png'
    }
    
    tabs_images = st.tabs(list(image_files.keys()))
    
    for idx, (name, filename) in enumerate(image_files.items()):
        with tabs_images[idx]:
            img_path = os.path.join(OUTPUTS_DIR, filename)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"{name} görüntüsü henüz oluşturulmadı. Lütfen önce eğitim yapın.")

# ============================================================================
# TAB 3: DETAYLI ANALİZ
# ============================================================================
with tab3:
    st.header("🔬 Detaylı Model Analizi")
    
    results_df, cv_df, pipeline_df = load_results()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Cross-Validation Sonuçları")
        if cv_df is not None:
            st.dataframe(
                cv_df.style.background_gradient(subset=['Mean F1'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # CV Box Plot
            fig = go.Figure()
            for _, row in cv_df.iterrows():
                fig.add_trace(go.Box(
                    y=[row['Min F1'], row['Mean F1'], row['Max F1']],
                    name=row['Model'],
                    boxmean='sd'
                ))
            fig.update_layout(title="Cross-Validation Score Dağılımı", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cross-validation sonuçları bekleniyor...")
    
    with col2:
        st.markdown("### ⚙️ Preprocessing Pipeline")
        if pipeline_df is not None:
            st.dataframe(pipeline_df, use_container_width=True)
            
            # Pipeline Visualization
            st.markdown("#### 🔄 İşlem Akışı")
            steps = [
                "1️⃣ Veri Yükleme",
                "2️⃣ Outlier Detection",
                "3️⃣ Scaling",
                "4️⃣ Class Balancing",
                "5️⃣ Train-Test Split"
            ]
            for step in steps:
                st.success(step)
        else:
            st.info("Pipeline bilgisi bekleniyor...")
    
    st.divider()
    
    # Model Karşılaştırma Radar Chart
    if results_df is not None:
        st.markdown("### 🎯 Model Karşılaştırma - Radar Chart")
        
        selected_model = st.selectbox(
            "Analiz edilecek modeli seçin:",
            results_df['Model'].unique()
        )
        
        model_data = results_df[results_df['Model'] == selected_model].iloc[0]
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            model_data['Test Accuracy'],
            model_data['Precision'],
            model_data['Recall'],
            model_data['F1-Score']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_model
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"{selected_model} - Performans Profili"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: VERİ İŞLEME
# ============================================================================
with tab4:
    st.header("📈 Veri İşleme ve EDA")
    
    preprocessing_images = {
        'Sınıf Dağılımı': 'class_distribution.png',
        'Outlier Tespiti': 'outlier_detection.png',
        'Sınıf Dengeleme': 'class_balancing_smote.png',
        'Korelasyon Matrisi': 'correlation_matrix.png',
        'Özellik Dağılımları': 'feature_distributions.png'
    }
    
    col1, col2 = st.columns(2)
    
    for idx, (name, filename) in enumerate(preprocessing_images.items()):
        img_path = os.path.join(OUTPUTS_DIR, filename)
        target_col = col1 if idx % 2 == 0 else col2
        
        with target_col:
            st.markdown(f"### {name}")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"{name} görüntüsü henüz oluşturulmadı.")
    
    # İstatistiksel Özet
    st.divider()
    st.markdown("### 📊 İstatistiksel Özet")
    
    summary_path = os.path.join(OUTPUTS_DIR, 'statistical_summary.csv')
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path, index_col=0)
        st.dataframe(summary_df.T, use_container_width=True, height=400)
    else:
        st.info("İstatistiksel özet henüz oluşturulmadı.")

# ============================================================================
# TAB 5: TOPLU İŞLEM
# ============================================================================
with tab5:
    st.header("⚙️ Toplu Tahmin İşlemi")
    
    st.markdown("""
    Bu bölümde birden fazla sample'ı aynı anda analiz edebilirsiniz.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Batch Ayarları")
        
        batch_size = st.slider("Batch Size:", 10, 500, 100, 10)
        
        if st.button("🚀 Toplu Analiz Başlat", use_container_width=True):
            with st.spinner("Toplu analiz yapılıyor..."):
                # Rastgele batch seç
                indices = np.random.choice(len(X_test), batch_size, replace=False)
                batch_X = X_test[indices]
                batch_y = y_test.iloc[indices]
                
                # Tahminler
                predictions = []
                confidences = []
                
                progress_bar = st.progress(0)
                for i, sample in enumerate(batch_X):
                    pred, conf = predict_sample(
                        model, dr_model, dr_method, 
                        sample.reshape(1, -1)
                    )
                    predictions.append(pred)
                    confidences.append(conf)
                    progress_bar.progress((i + 1) / batch_size)
                
                # Sonuçları kaydet
                st.session_state['batch_results'] = {
                    'predictions': predictions,
                    'true_labels': batch_y.values,
                    'confidences': confidences,
                    'indices': indices
                }
                
                st.success("✅ Toplu analiz tamamlandı!")
    
    with col2:
        st.markdown("### 📊 Batch Sonuçları")
        
        if 'batch_results' in st.session_state:
            results = st.session_state['batch_results']
            
            # Metrikler
            predictions = np.array(results['predictions'])
            true_labels = np.array(results['true_labels'])
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            acc = accuracy_score(true_labels, predictions)
            prec = precision_score(true_labels, predictions)
            rec = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{acc:.3f}")
            col_m2.metric("Precision", f"{prec:.3f}")
            col_m3.metric("Recall", f"{rec:.3f}")
            col_m4.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, predictions)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Attack'],
                y=['Normal', 'Attack'],
                colorscale='RdYlGn_r',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            fig.update_layout(
                title="Batch Confusion Matrix",
                xaxis_title="Tahmin",
                yaxis_title="Gerçek",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sonuç Tablosu
            st.markdown("#### 📋 Detaylı Sonuçlar")
            results_table = pd.DataFrame({
                'Index': results['indices'],
                'True Label': ['Normal' if x==0 else 'Attack' for x in true_labels],
                'Prediction': ['Normal' if x==0 else 'Attack' for x in predictions],
                'Confidence': [f"{x:.3f}" for x in results['confidences']],
                'Correct': ['✅' if p==t else '❌' for p, t in zip(predictions, true_labels)]
            })
            
            st.dataframe(results_table, use_container_width=True, height=400)
            
            # CSV İndirme
            csv = results_table.to_csv(index=False)
            st.download_button(
                label="📥 Sonuçları İndir (CSV)",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("👈 Toplu analiz başlatmak için sol paneli kullanın")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>Advanced Network Intrusion Detection System</strong></p>
    <p>Gazi Üniversitesi - Bilgisayar Mühendisliği</p>
    <p>Developed by Canberk Aykanat</p>
</div>
""", unsafe_allow_html=True)