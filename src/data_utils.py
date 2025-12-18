import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from scipy import stats
import os

OUTPUTS_DIR = 'outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_raw_data():
    """Veri setini yükler"""
    print("="*80)
    print("VERİ SETİ YÜKLEME")
    print("="*80)
    
    kdd_data = fetch_kddcup99(subset='SA', percent10=True, as_frame=True)
    df = kdd_data.frame
    
    print(f"\nVeri seti boyutu: {df.shape}")
    print(f"Öznitelik sayısı: {df.shape[1] - 1}")
    print(f"Örnek sayısı: {df.shape[0]}")
    
    # Binary classification: Normal vs Attack
    df['label'] = df['labels'].apply(lambda x: 0 if x == b'normal.' or x == 'normal.' else 1)
    df = df.drop('labels', axis=1)
    
    # Kategorik değişkenleri encode et
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in categorical_cols:
        categorical_cols.remove('label')
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def analyze_class_distribution(df):
    """Sınıf dağılımını analiz eder ve görselleştirir"""
    print("\n" + "="*80)
    print("SINIF DAĞILIMI ANALİZİ")
    print("="*80)
    
    class_counts = df['label'].value_counts()
    print(f"\nNormal (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"Attack (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
    print(f"Dengesizlik Oranı: 1:{class_counts[0]/class_counts[1]:.2f}")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    class_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sınıf (0: Normal, 1: Attack)')
    axes[0].set_ylabel('Örnek Sayısı')
    axes[0].set_xticklabels(['Normal', 'Attack'], rotation=0)
    
    # Pie chart
    axes[1].pie(class_counts, labels=['Normal', 'Attack'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Sınıf Oranları', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return class_counts

def detect_outliers(df, method='zscore', threshold=3):
    """Outlier detection uygular"""
    print("\n" + "="*80)
    print(f"OUTLIER TESPİTİ ({method.upper()})")
    print("="*80)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(X))
        outliers = (z_scores > threshold).any(axis=1)
    elif method == 'iqr':
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    elif method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(X) == -1
    else:
        raise ValueError("method must be 'zscore', 'iqr', or 'isolation_forest'")
    
    outlier_count = outliers.sum()
    outlier_percentage = (outlier_count / len(df)) * 100
    
    print(f"\nToplam Outlier: {outlier_count} ({outlier_percentage:.2f}%)")
    print(f"Temiz Veri: {len(df) - outlier_count} ({100-outlier_percentage:.2f}%)")
    
    # Outlier'ları kaldır
    df_clean = df[~outliers].copy()
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Outlier sayıları
    outlier_data = pd.DataFrame({
        'Kategori': ['Temiz Veri', 'Outlier'],
        'Sayı': [len(df) - outlier_count, outlier_count]
    })
    axes[0].bar(outlier_data['Kategori'], outlier_data['Sayı'], color=['#3498db', '#e74c3c'])
    axes[0].set_title(f'Outlier Tespiti - {method.upper()}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Örnek Sayısı')
    
    # İlk 5 özellik için box plot
    sample_features = X.columns[:5]
    X[sample_features].boxplot(ax=axes[1])
    axes[1].set_title('Özellik Dağılımları (İlk 5)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Değer')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'outlier_detection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_clean

def apply_scaling(X_train, X_test, method='standard'):
    print(f"\n{method.upper()} SCALING uygulanıyor...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def balance_dataset(X_train, y_train, method='smote'):
    print(f"SINIF DENGELENMESİ ({method.upper()})")
    
    original_counts = pd.Series(y_train).value_counts()
    print(f"\nOrijinal Dağılım:")
    print(f"Normal (0): {original_counts[0]}")
    print(f"Attack (1): {original_counts[1]}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif method == 'random_over':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'random_under':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    else:
        raise ValueError("Invalid balancing method")
    
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    new_counts = pd.Series(y_balanced).value_counts()
    print(f"\nDengelenmiş Dağılım:")
    print(f"Normal (0): {new_counts[0]}")
    print(f"Attack (1): {new_counts[1]}")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Öncesi
    original_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Dengeleme Öncesi', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sınıf')
    axes[0].set_ylabel('Örnek Sayısı')
    axes[0].set_xticklabels(['Normal', 'Attack'], rotation=0)
    
    # Sonrası
    new_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
    axes[1].set_title(f'Dengeleme Sonrası ({method.upper()})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sınıf')
    axes[1].set_ylabel('Örnek Sayısı')
    axes[1].set_xticklabels(['Normal', 'Attack'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f'class_balancing_{method}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return X_balanced, y_balanced

def create_eda_report(df):
    print("EXPLORATORY DATA ANALYSIS")
    
    # 1. Korelasyon matrisi
    plt.figure(figsize=(12, 10))
    corr_matrix = df.drop('label', axis=1).corr()
    sns.heatmap(corr_matrix.iloc[:15, :15], annot=False, cmap='coolwarm', center=0)
    plt.title('Korelasyon Matrisi (İlk 15 Özellik)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Özellik dağılımları
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(df.columns[:9]):
        if col != 'label':
            axes[idx].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}', fontsize=10)
            axes[idx].set_ylabel('Frekans')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. İstatistiksel özet
    summary_stats = df.describe()
    summary_stats.to_csv(os.path.join(OUTPUTS_DIR, 'statistical_summary.csv'))
    
    print("\nEDA raporu oluşturuldu ve kaydedildi.")

def load_and_preprocess_data(outlier_method='zscore', scaling_method='standard', 
                            balance_method='smote', apply_balancing=True):
    
    # 1. Veri Yükleme
    df = load_raw_data()
    
    # 2. EDA
    create_eda_report(df)
    
    # 3. Sınıf Dağılımı Analizi
    analyze_class_distribution(df)
    
    # 4. Outlier Detection
    df_clean = detect_outliers(df, method=outlier_method)
    
    # 5. Train-Test Split
    X = df_clean.drop('label', axis=1)
    y = df_clean['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Scaling
    X_train_scaled, X_test_scaled, scaler = apply_scaling(
        X_train, X_test, method=scaling_method
    )
    
    # 7. Class Balancing (sadece train set'e uygulanır)
    if apply_balancing:
        X_train_balanced, y_train_balanced = balance_dataset(
            X_train_scaled, y_train, method=balance_method
        )
    else:
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    print("\n" + "="*80)
    print("VERİ ÖN İŞLEME TAMAMLANDI")
    print("="*80)
    print(f"Eğitim Seti: {X_train_balanced.shape}")
    print(f"Test Seti: {X_test_scaled.shape}")
    
    # Pipeline bilgilerini kaydet
    pipeline_info = {
        'outlier_method': outlier_method,
        'scaling_method': scaling_method,
        'balance_method': balance_method if apply_balancing else 'None',
        'train_samples': X_train_balanced.shape[0],
        'test_samples': X_test_scaled.shape[0],
        'n_features': X_train_balanced.shape[1]
    }
    
    pd.DataFrame([pipeline_info]).to_csv(
        os.path.join(OUTPUTS_DIR, 'preprocessing_pipeline.csv'), index=False
    )
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, X.columns