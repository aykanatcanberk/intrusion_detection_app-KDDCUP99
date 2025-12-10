import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

from src import data_utils
from src import model_utils

ARTIFACTS_DIR = 'artifacts'
OUTPUTS_DIR = 'outputs'

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def main():
    # 1. Veri Yükleme
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _ = data_utils.load_and_preprocess_data()

    # 2. Boyut İndirgeme
    print("\n" + "="*80)
    print("BOYUT İNDİRGEME YÖNTEMLERİ ")
    print("="*80)
    dim_reduction_results, dr_models = model_utils.apply_dim_reduction(X_train_scaled, X_test_scaled, y_train)

    # 3. Model Eğitimi
    print("\nMAKİNE ÖĞRENMESİ MODELLERİ")
    print("="*80)
    
    models = model_utils.get_models()
    results = []

    # A) Orijinal Veri ile Eğitim
    print("ORİJİNAL VERİ SETİ İLE MODEL EĞİTİMİ")
    print("-" * 80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Fark (Overfit)':<15}")
    print("-" * 80)

    for model_name, model in models.items():
        start_time = time.time()
        
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred) # Eğitim Başarısı
        test_acc = accuracy_score(y_test, y_test_pred)    # Test Başarısı
        
        train_time = time.time() - start_time
        
        prec = precision_score(y_test, y_test_pred, average='binary')
        rec = recall_score(y_test, y_test_pred, average='binary')
        f1 = f1_score(y_test, y_test_pred, average='binary')
        
        diff = train_acc - test_acc 
        
        results.append({
            'Model': model_name,
            'Dimension Reduction': 'Original',
            'Accuracy': test_acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Training Time': train_time
        })
        
        print(f"{model_name:<20} {train_acc:.4f}       {test_acc:.4f}       {diff:.4f}")

    # B) Boyut İndirgeme ile Eğitim
    for dr_method, dr_data in dim_reduction_results.items():
        print("\n" + "-"*80)
        print(f"{dr_method} İLE MODEL EĞİTİMİ")
        print("-" * 80)
        print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Fark (Overfit)':<15}")
        print("-" * 80)
        
        X_train_dr = dr_data['X_train']
        X_test_dr = dr_data['X_test']
        
        for model_name, model in models.items():
            start_time = time.time()
            
            current_models = model_utils.get_models()
            current_model = current_models[model_name]
            
            current_model.fit(X_train_dr, y_train)
            
            y_train_pred = current_model.predict(X_train_dr)
            y_test_pred = current_model.predict(X_test_dr)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            train_time = time.time() - start_time
            
            prec = precision_score(y_test, y_test_pred, average='binary')
            rec = recall_score(y_test, y_test_pred, average='binary')
            f1 = f1_score(y_test, y_test_pred, average='binary')
            
            diff = train_acc - test_acc
            
            results.append({
                'Model': model_name,
                'Dimension Reduction': dr_method,
                'Accuracy': test_acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'Training Time': train_time
            })
            print(f"{model_name:<20} {train_acc:.4f}       {test_acc:.4f}       {diff:.4f}")

    # 4. Sonuç Analizi
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['F1-Score'].idxmax()]

    print("\n" + "="*80)
    print("EN İYİ PERFORMANS GÖSTEREN MODEL")
    print("="*80)
    print(f"Model: {best_result['Model']}")
    print(f"Boyut İndirgeme: {best_result['Dimension Reduction']}")
    print(f"Accuracy: {best_result['Accuracy']:.4f}")
    print(f"F1-Score: {best_result['F1-Score']:.4f}")

    # 5. Görselleştirme (OUTPUTS klasörüne kaydet)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    pivot_df = results_df.pivot(index='Model', columns='Dimension Reduction', values='F1-Score')
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('F1-Score Karşılaştırması')
    plt.legend(title='Dim Reduction', bbox_to_anchor=(1.05, 1))

    plt.subplot(2, 2, 2)
    pivot_df = results_df.pivot(index='Model', columns='Dimension Reduction', values='Accuracy')
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Accuracy Karşılaştırması')
    plt.legend(title='Dim Reduction', bbox_to_anchor=(1.05, 1))

    plt.subplot(2, 2, 3)
    pivot_df = results_df.pivot(index='Model', columns='Dimension Reduction', values='Training Time')
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Eğitim Süresi Karşılaştırması')
    plt.legend(title='Dim Reduction', bbox_to_anchor=(1.05, 1))

    plt.subplot(2, 2, 4)
    model_avg = results_df.groupby('Model')['F1-Score'].mean().sort_values()
    model_avg.plot(kind='barh')
    plt.title('Modellerin Ortalama F1-Score Performansı')
    plt.tight_layout()

    comp_plot_path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGörselleştirme kaydedildi: {comp_plot_path}")

    # 6. En İyi Modeli Kaydetme (ARTIFACTS klasörüne)
    best_model_name = best_result['Model']
    best_dr_method = best_result['Dimension Reduction']
    
    final_models = model_utils.get_models()
    final_model = final_models[best_model_name]

    if best_dr_method == 'Original':
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = dim_reduction_results[best_dr_method]['X_train']
        X_test_final = dim_reduction_results[best_dr_method]['X_test']

    final_model.fit(X_train_final, y_train)

    # Dosya yolları
    model_path = os.path.join(ARTIFACTS_DIR, 'best_intrusion_detection_model.pkl')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Boyut indirgeme modelini kaydet
    if best_dr_method != 'Original':
        dr_model_to_save = dr_models[best_dr_method]
        if best_dr_method == 'Autoencoder':
             dr_path = os.path.join(ARTIFACTS_DIR, 'dim_reduction_model.h5')
             dr_model_to_save.save(dr_path)
        else:
            dr_path = os.path.join(ARTIFACTS_DIR, 'dim_reduction_model.pkl')
            with open(dr_path, 'wb') as f:
                pickle.dump(dr_model_to_save, f)
        print(f"Boyut indirgeme modeli kaydedildi: {dr_path}")

    # Model bilgilerini metin dosyasına yaz (OUTPUTS klasörüne)
    info_path = os.path.join(OUTPUTS_DIR, 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"{best_model_name}\n")
        f.write(f"{best_dr_method}")

    print(f"Model ve Scaler kaydedildi: {ARTIFACTS_DIR}")

    # Final Değerlendirme & Confusion Matrix
    y_pred_final = final_model.predict(X_test_final)
    print("\n" + "="*80)
    print("FİNAL MODEL PERFORMANSI (Test Seti)")
    print("="*80)
    print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Attack']))

    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name} with {best_dr_method}')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    
    cm_path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion Matrix kaydedildi: {cm_path}")

if __name__ == "__main__":
    main()