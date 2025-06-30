import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import random
import time
import math
import warnings

# Ignorar warnings para mayor claridad en la salida
warnings.filterwarnings('ignore')

# --- 1. Carga y Pre-procesamiento del Dataset ---

def load_and_preprocess_data(train_path, test_path):
    print("Cargando datos...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datos cargados.")

    # Combinar para asegurar el mismo pre-procesamiento y obtener el esquema completo de columnas
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Identificar características y la variable objetivo (label binaria)
    # Según el paper, el label binario es 'label' (0 para normal, 1 para ataque)
    # Eliminamos 'id' y 'attack_cat' ya que 'label' es la objetivo binaria
    X = combined_df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = combined_df['label']

    print(f"Número de características originales antes del pre-procesamiento: {X.shape[1]}")

    # Identificar columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Crear preprocesador
    # One-Hot Encoding para categóricas, Min-Max Scaling para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    print("Aplicando pre-procesamiento...")
    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)
    print(f"Dimensiones de los datos pre-procesados: {X_processed.shape}")
    print(f"Número de características después del pre-procesamiento (espacio de búsqueda para la metaheurística): {X_processed.shape[1]}")

    # Separar de nuevo en conjuntos de entrenamiento y prueba usando los índices originales
    # Asumiendo que los archivos train_df y test_df originales mantenían el orden
    train_size = len(train_df)
    X_train_processed = X_processed[:train_size]
    X_test_processed = X_processed[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("Datos pre-procesados y separados (train/test).")

    return X_train_processed, X_test_processed, y_train, y_test, X.columns, categorical_features

# --- 2. Definición de la Fitness Function (Wrapper) ---

def fitness_function(individual, X_train, X_test, y_train, y_test):
    # Individual es un vector binario (numpy array)
    selected_features_indices = np.where(individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features = len(individual) # Longitud del vector binario = número de features en el espacio optimizado

    # Penalizar si no se selecciona ninguna característica o si se seleccionan todas
    if num_selected_features == 0 or num_selected_features == total_features:
         return -1.0 # Fitness muy bajo para selecciones inválidas

    # Seleccionar solo las características indicadas del dataset pre-procesado
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Entrenar y evaluar el clasificador (Random Forest)
    # Usamos parámetros de RF para evaluación rápida dentro del fitness
    classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)

    # Calcular la métrica de rendimiento (Accuracy)
    accuracy = accuracy_score(y_test, y_pred)

    # Calcular el fitness (Maximizar Accuracy, Minimizar Features)
    # w1=1, w2=0.05 (penalización pequeña por característica)
    w1 = 1.0
    w2 = 0.05
    fitness = w1 * accuracy - w2 * (num_selected_features / total_features)

    return fitness

# --- 3. Implementación del Algoritmo Simulated Annealing (SA) ---

def acceptance_probability(current_fitness, neighbor_fitness, temperature):
    # Si el nuevo estado es mejor, siempre aceptarlo
    if neighbor_fitness > current_fitness:
        return 1.0
    # Si el nuevo estado es peor, calcular la probabilidad de aceptación
    # Usamos exp(delta_fitness / T) donde delta_fitness es neighbor_fitness - current_fitness (será negativo)
    # Para evitar underflow/overflow con exp
    try:
       return math.exp((neighbor_fitness - current_fitness) / temperature)
    except OverflowError:
       return 0.0 # Para delta_fitness muy negativo / T -> -inf, exp -> 0

def run_simulated_annealing(X_train, X_test, y_train, y_test, T_initial, alpha, num_iterations, total_features):
    print(f"\nEjecutando Simulated Annealing (SA) con T_initial={T_initial}, alpha={alpha}, {num_iterations} iteraciones...")
    print(f"Espacio de búsqueda binario tiene {total_features} dimensiones.")

    # Inicialización: Generar una solución inicial aleatoria válida
    current_individual = np.random.randint(0, 2, size=total_features)
    while np.sum(current_individual) == 0 or np.sum(current_individual) == total_features:
         current_individual = np.random.randint(0, 2, size=total_features)

    current_fitness = fitness_function(current_individual, X_train, X_test, y_train, y_test)

    # Mantener un seguimiento de la mejor solución encontrada hasta ahora
    best_individual = current_individual.copy()
    best_fitness = current_fitness

    history_best_fitness = [best_fitness] # Para graficar la convergencia

    temperature = T_initial
    start_time = time.time()

    for i in range(num_iterations):
        # Generar una solución vecina (aplicar operador de vecindario)
        # Crear una copia para no modificar el individuo actual directamente
        neighbor_individual = current_individual.copy()

        # Invertir un bit aleatorio
        flip_index = random.randint(0, total_features - 1)
        neighbor_individual[flip_index] = 1 - neighbor_individual[flip_index] # Flip the bit (0 to 1 or 1 to 0)

        # Asegurar que la solución vecina sea válida (no todo 0s o todo 1s)
        # Si el flip resultó en un vector inválido, revertimos el flip
        if np.sum(neighbor_individual) == 0 or np.sum(neighbor_individual) == total_features:
             neighbor_individual[flip_index] = 1 - neighbor_individual[flip_index] # Revertir el flip
             # Podríamos intentar otro flip, pero para mantener el paso simple,
             # si revertir no funciona (ej. si flip_index es la única feature seleccionada/no seleccionada),
             # el individuo vecino será igual al actual, y el acceptance_probability lo manejará (exp(0/T)=1).
             # Es mejor recalcular el fitness aquí por si acaso.
             neighbor_fitness = fitness_function(neighbor_individual, X_train, X_test, y_train, y_test)
             # Si el vecino resultante sigue siendo inválido después de revertir (solo pasa si el original era inválido)
             if neighbor_fitness <= -0.9: # Usamos -0.9 como umbral para nuestro -1.0 de penalización
                 # Si el vecino inválido fue generado del inválido original, intentamos una regeneración aleatoria
                 neighbor_individual = np.random.randint(0, 2, size=total_features)
                 while np.sum(neighbor_individual) == 0 or np.sum(neighbor_individual) == total_features:
                      neighbor_individual = np.random.randint(0, 2, size=total_features)
                 neighbor_fitness = fitness_function(neighbor_individual, X_train, X_test, y_train, y_test)


        else:
            # Calcular el fitness del vecino si es válido
            neighbor_fitness = fitness_function(neighbor_individual, X_train, X_test, y_train, y_test)


        # Decidir si aceptar la nueva solución (criterio de Metropolis)
        ap = acceptance_probability(current_fitness, neighbor_fitness, temperature)

        if ap > random.random():
            current_individual = neighbor_individual.copy()
            current_fitness = neighbor_fitness

        # Actualizar la mejor solución encontrada hasta ahora (si la solución *actual* es mejor)
        # Nota: La solución actual puede ser la anterior o la vecina aceptada
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = current_individual.copy()

        history_best_fitness.append(best_fitness) # Registrar el mejor fitness hasta esta iteración

        # Enfriar la temperatura
        temperature = temperature * alpha

        # Opcional: Imprimir progreso periódicamente
        if (i + 1) % 100 == 0 or i == num_iterations - 1:
             elapsed_time = time.time() - start_time
             print(f"Iteración {i+1}/{num_iterations}, Temp: {temperature:.4f}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}, Tiempo: {elapsed_time:.2f}s")

    end_time = time.time()
    print("\nSimulated Annealing finalizado.")
    print(f"Tiempo total de ejecución SA: {end_time - start_time:.2f}s")

    return best_individual, best_fitness, history_best_fitness

# --- 4. Evaluación del Mejor Resultado Final ---

def evaluate_final_model(best_individual, X_train, X_test, y_train, y_test, original_features, cat_feature_names):
    selected_features_indices = np.where(best_individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features_in_optimized_space = len(best_individual) # Es el tamaño del espacio de búsqueda binario


    print("\n--- Evaluación del Mejor Subconjunto de Características (Simulated Annealing) ---")
    print(f"Características seleccionadas: {num_selected_features}/{total_features_in_optimized_space} (del espacio pre-procesado)")

    if num_selected_features == 0:
        print("No se seleccionó ninguna característica. No se puede evaluar el modelo final.")
        return

    # Seleccionar solo las características del mejor individuo
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Entrenar el clasificador final con los parámetros tunneados del paper (Table 6)
    # Usaremos los valores de la Table 6 del paper para RF
    final_classifier = RandomForestClassifier(n_estimators=500, # Más robusto que en fitness
                                            max_depth=4,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            random_state=42,
                                            n_jobs=-1)

    print("Entrenando clasificador final con el mejor subconjunto...")
    final_classifier.fit(X_train_selected, y_train)
    print("Evaluando clasificador final en el test set...")
    y_pred = final_classifier.predict(X_test_selected)

    # Calcular métricas de evaluación
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Para métricas como DR, PR, F1, FPR necesitamos TP, TN, FP, FN
    # En clasificación binaria (0, 1):
    # TN: Actual 0, Predicted 0
    # FP: Actual 0, Predicted 1
    # FN: Actual 1, Predicted 0
    # TP: Actual 1, Predicted 1

    # Evitar errores si la matriz de confusión no tiene la forma esperada (ej. si solo hay una clase en el test set, poco probable aquí)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Esto puede ocurrir si una clase no aparece en y_true o y_pred
        print("Advertencia: Matriz de confusión no es 2x2. Recalculando métricas.")
        TP = ((y_test == 1) & (y_pred == 1)).sum()
        TN = ((y_test == 0) & (y_pred == 0)).sum()
        FP = ((y_test == 0) & (y_pred == 1)).sum()
        FN = ((y_test == 1) & (y_pred == 0)).sum()


    dr = recall_score(y_test, y_pred, average='binary') # DR (Detection Rate) = Recall
    pr = precision_score(y_test, y_pred, average='binary') # PR (Precision)
    f1 = f1_score(y_test, y_pred, average='binary')     # F1-Score

    # FPR (False Positive Rate) = FP / (FP + TN)
    # Manejar división por cero si TN + FP = 0 (caso raro en datasets grandes)
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    print("\nMétricas de rendimiento finales:")
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Detection Rate (DR): {dr:.4f}")
    print(f"Precision (PR): {pr:.4f}")
    print(f"F1-Score (F1): {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Número de características seleccionadas: {num_selected_features}")
    print(f"Índices de las características seleccionadas (en el espacio pre-procesado): {selected_features_indices.tolist()}")


# --- Main Execution ---

if __name__ == "__main__":
    # IMPORTANT: Download the datasets first
    # UNSW-NB15_training-set.csv and UNSW_NB15_testing-set.csv
    # You can find them online, e.g., on Kaggle or the UNSW website
    # Descarga los archivos y actualiza las rutas:
    train_file = 'UNSW_NB15_training-set.csv' # <--- ACTUALIZA ESTA RUTA
    test_file = 'UNSW_NB15_testing-set.csv'   # <--- ACTUALIZA ESTA RUTA

    # Verificar si los archivos existen
    import os
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"ERROR: Archivos de dataset no encontrados en '{train_file}' y '{test_file}'.")
        print("Por favor, descarga los datasets UNSW-NB15 (training-set.csv y testing-set.csv)")
        print("y actualiza las rutas de los archivos en el código.")
    else:
        # Carga y pre-procesamiento
        X_train, X_test, y_train, y_test, original_features, categorical_features = load_and_preprocess_data(train_file, test_file)

        # Número total de características después del pre-procesamiento
        total_features_count = X_train.shape[1]

        # Parámetros para Simulated Annealing (ajustar según rendimiento, estos son ejemplos iniciales)
        # T_initial: Temperatura inicial alta.
        # alpha: Factor de enfriamiento (cercano a 1, ej 0.99 o 0.995).
        # num_iterations: Número de pasos de enfriamiento (cuántos "vecinos" probamos en total).
        # Estos valores requieren experimentación para optimizar.
        sa_T_initial = 100.0
        sa_alpha = 0.995 # Enfriamiento lento para explorar más
        sa_num_iterations = 5000 # Ejecutar más iteraciones que generaciones en BDE, ya que es trayectoria


        # Ejecutar Simulated Annealing para encontrar el mejor subconjunto de características
        best_features_binary_vector, final_best_fitness, fitness_history = run_simulated_annealing(
            X_train, X_test, y_train, y_test,
            sa_T_initial, sa_alpha, sa_num_iterations,
            total_features_count
        )

        # Evaluar el modelo final con el mejor subconjunto
        evaluate_final_model(best_features_binary_vector, X_train, X_test, y_train, y_test, original_features, categorical_features)

        # Opcional: Graficar la convergencia del fitness (requiere matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(fitness_history)
            plt.xlabel('Iteración')
            plt.ylabel('Mejor Fitness Encontrado Hasta Ahora (Accuracy Penalizada)')
            plt.title('Convergencia de Simulated Annealing')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("\nInstala matplotlib para ver el gráfico de convergencia: pip install matplotlib")