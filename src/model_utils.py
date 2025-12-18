import time
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import LocallyLinearEmbedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Makine öğrenmesi algoritmaları
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def apply_dim_reduction(X_train_scaled, X_test_scaled, y_train, n_components=20):

    dim_reduction_results = {}
    dr_models = {} 

    # 1. PCA
    print("\n1. PCA uygulanıyor...")
    start_time = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    pca_time = time.time() - start_time
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA tamamlandı. Süre: {pca_time:.2f} saniye")
    print(f"Açıklanan Varyans: %{explained_variance:.2f}")
    
    dim_reduction_results['PCA'] = {
        'X_train': X_train_pca, 
        'X_test': X_test_pca, 
        'time': pca_time
    }
    dr_models['PCA'] = pca

    # 2. ICA
    print("\n2. ICA uygulanıyor...")
    start_time = time.time()
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    X_train_ica = ica.fit_transform(X_train_scaled)
    X_test_ica = ica.transform(X_test_scaled)
    ica_time = time.time() - start_time
    
    print(f"ICA tamamlandı. Süre: {ica_time:.2f} saniye")
    
    dim_reduction_results['ICA'] = {
        'X_train': X_train_ica, 
        'X_test': X_test_ica, 
        'time': ica_time
    }
    dr_models['ICA'] = ica

    # 3. LDA
    print("\n3. LDA uygulanıyor...")
    start_time = time.time()
    lda_components = min(n_components, len(np.unique(y_train)) - 1)
    lda = LDA(n_components=lda_components)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    lda_time = time.time() - start_time
    
    print(f"LDA tamamlandı. Süre: {lda_time:.2f} saniye")
    print(f"Kullanılan komponent sayısı: {lda_components}")
    
    dim_reduction_results['LDA'] = {
        'X_train': X_train_lda, 
        'X_test': X_test_lda, 
        'time': lda_time
    }
    dr_models['LDA'] = lda

    # 4. Autoencoder
    print("\n4. Autoencoder uygulanıyor...")
    start_time = time.time()
    input_dim = X_train_scaled.shape[1]
    encoding_dim = n_components

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)

    decoder = Dense(64, activation='relu')(encoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)

    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled, 
        epochs=15, 
        batch_size=256, 
        shuffle=True, 
        validation_split=0.1, 
        verbose=0
    )

    X_train_ae = encoder_model.predict(X_train_scaled, verbose=0)
    X_test_ae = encoder_model.predict(X_test_scaled, verbose=0)
    ae_time = time.time() - start_time
    
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Autoencoder tamamlandı. Süre: {ae_time:.2f} saniye")
    print(f"Final Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")
    
    dim_reduction_results['Autoencoder'] = {
        'X_train': X_train_ae, 
        'X_test': X_test_ae, 
        'time': ae_time
    }
    dr_models['Autoencoder'] = encoder_model
        
    return dim_reduction_results, dr_models

def get_models():
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'ANN': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
    }

