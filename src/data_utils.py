import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    print("VERİ SETİ YÜKLEME VE ÖN İŞLEME")
    
    kdd_data = fetch_kddcup99(subset='SA', percent10=True, as_frame=True)
    df = kdd_data.frame

    print(f"\nVeri seti boyutu: {df.shape}")
    print(f"Öznitelik sayısı: {df.shape[1] - 1}")
    print(f"Örnek sayısı: {df.shape[0]}")

    # (Binary: Normal vs Attack)
    df['label'] = df['labels'].apply(lambda x: 0 if x == b'normal.' or x == 'normal.' else 1)
    
    # Orijinal labels sütununu kaldır
    df = df.drop('labels', axis=1)

    # Kategorik değişkenleri encode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in categorical_cols:
        categorical_cols.remove('label')

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nEğitim seti boyutu: {X_train_scaled.shape}")
    print(f"Test seti boyutu: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df.columns