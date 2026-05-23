import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# BLOQUE 1: CARGA E INGESTA
# ==========================================

df = pd.read_csv('hmeq.csv')
print("Dataset cargado correctamente.")

# ==========================================
# BLOQUE 2: PREPROCESAMIENTO
# ==========================================

# 1. Tratamiento de nulos: Imputación por mediana (numéricos) y moda (categóricos)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = ['REASON', 'JOB']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2. Ingeniería de características 
# Justificación: El ratio es más predictivo que las variables por separado
df['LTV'] = df['LOAN'] / df['VALUE']
# Otra variable: Cuotas de crédito respecto a la edad del historial
df['RISK_RATIO'] = df['DEBTINC'] / (df['CLAGE'] + 1)

# 3. Codificación de variables cualitativas
df = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)

# 4. Preparación de Tensores
X = df.drop('BAD', axis=1)
y = df['BAD']

# 5. Escalado (Estandarización para que todas las variables pesen igual)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. División Entrenamiento / Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Dimensiones del tensor de entrada: {X_train.shape}")

# ==========================================
# BLOQUE 3: MODELADO (Arquitectura MLP)
# ==========================================
model = Sequential([
    # Capa de entrada y primera capa oculta con ReLU
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2), # Regularización
    
    # Segunda capa oculta
    Dense(16, activation='relu'),
    
    # Capa de salida con Sigmoide (genera probabilidad entre 0 y 1)
    Dense(1, activation='sigmoid')
])

# Compilación con Binary Cross-Entropy (obligatoria para clasificación binaria)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.Recall(), 'accuracy'])

# Early Stopping para evitar el sobreajuste (overfitting)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamiento del modelo
print("\nIniciando entrenamiento...")
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    validation_split=0.2, 
                    callbacks=[early_stop], 
                    batch_size=32, 
                    verbose=1)

# ==========================================
# BLOQUE 4: EVALUACIÓN CRÍTICA
# ==========================================

# Predicciones
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.3).astype(int)

print("\n--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))

# Visualización de la convergencia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss Entreno')
plt.plot(history.history['val_loss'], label='Loss Val')
plt.title('Evolución de la Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acc Entreno')
plt.plot(history.history['val_accuracy'], label='Acc Val')
plt.title('Evolución de la Precisión')
plt.legend()
plt.show()