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


warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path):
    print("Cargando datos...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datos cargados.")


    combined_df = pd.concat([train_df, test_df], ignore_index=True)


    X = combined_df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = combined_df['label']

    print(f"Número de características originales antes del pre-procesamiento: {X.shape[1]}")



    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    print("Aplicando pre-procesamiento...")

    X_processed = preprocessor.fit_transform(X)
    print(f"Dimensiones de los datos pre-procesados: {X_processed.shape}")
    print(f"Número de características después del pre-procesamiento (espacio de búsqueda para la metaheurística): {X_processed.shape[1]}")


    train_size = len(train_df)
    X_train_processed = X_processed[:train_size]
    X_test_processed = X_processed[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("Datos pre-procesados y separados (train/test).")


    return X_train_processed, X_test_processed, y_train, y_test, X.columns, categorical_features 



def fitness_function(individual, X_train, X_test, y_train, y_test):

    selected_features_indices = np.where(individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features = len(individual) 

    if num_selected_features == 0 or num_selected_features == total_features:
         return -1.0 

    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]


    classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)


    accuracy = accuracy_score(y_test, y_pred)

    if num_selected_features > 0 and num_selected_features < total_features:
        w1 = 1.0
        w2 = 0.05
        fitness = w1 * accuracy - w2 * (num_selected_features / total_features)
    else:
        fitness = -1.0 

    return fitness

def sigmoid(x):

    if x >= 500:
        return 1.0
    elif x <= -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def run_bde(X_train, X_test, y_train, y_test, pop_size, num_generations, F, CR, total_features):
    print(f"\nEjecutando Binary Differential Evolution (BDE) con {pop_size} individuos por {num_generations} generaciones...")
    print(f"Espacio de búsqueda binario tiene {total_features} dimensiones.")


    population = np.random.randint(0, 2, size=(pop_size, total_features))

    for i in range(pop_size):
        while np.sum(population[i]) == 0 or np.sum(population[i]) == total_features:
             population[i] = np.random.randint(0, 2, size=total_features)


    fitness_scores = np.array([fitness_function(ind, X_train, X_test, y_train, y_test) for ind in population])

    best_individual = population[np.argmax(fitness_scores)].copy()
    best_fitness = np.max(fitness_scores)
    history_best_fitness = [best_fitness]

    start_time = time.time()

    for gen in range(num_generations):
        new_population = np.zeros((pop_size, total_features), dtype=int)
        new_fitness_scores = np.zeros(pop_size)

        for i in range(pop_size):
        
            indices = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = random.sample(indices, 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]


            trial_vector = np.zeros(total_features)
 
            jrand = random.randint(0, total_features - 1)

            for j in range(total_features):
   
                if random.random() < CR or j == jrand:

     
                    v_j = sigmoid(population[i, j] + F * (b[j] - c[j])) 

                    if random.random() < v_j:
                         trial_vector[j] = 1
                    else:
                         trial_vector[j] = 0
                else:
                    trial_vector[j] = population[i, j] 

            if np.sum(trial_vector) == 0 or np.sum(trial_vector) == total_features:

                if np.sum(population[i]) > 0 and np.sum(population[i]) < total_features:
                     trial_vector = population[i].copy() # Usar el padre si era válido
                else:

                     trial_vector = np.random.randint(0, 2, size=total_features)
                     while np.sum(trial_vector) == 0 or np.sum(trial_vector) == total_features:
                         trial_vector = np.random.randint(0, 2, size=total_features)



            trial_fitness = fitness_function(trial_vector, X_train, X_test, y_train, y_test)


            if trial_fitness > fitness_scores[i]:
                new_population[i] = trial_vector
                new_fitness_scores[i] = trial_fitness
            else:
                new_population[i] = population[i]
                new_fitness_scores[i] = fitness_scores[i]

        population = new_population
        fitness_scores = new_fitness_scores

        current_best_fitness = np.max(fitness_scores)
        current_best_individual = population[np.argmax(fitness_scores)].copy()

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()

        history_best_fitness.append(best_fitness)

        if (gen + 1) % 10 == 0 or gen == num_generations - 1:
             elapsed_time = time.time() - start_time
             print(f"Generación {gen+1}/{num_generations}, Mejor Fitness: {best_fitness:.4f}, Tiempo: {elapsed_time:.2f}s")

    end_time = time.time()
    print("\nBDE finalizado.")
    print(f"Tiempo total de ejecución BDE: {end_time - start_time:.2f}s")

    return best_individual, best_fitness, history_best_fitness


def evaluate_final_model(best_individual, X_train, X_test, y_train, y_test, original_features, cat_feature_names):
    selected_features_indices = np.where(best_individual == 1)[0]
    num_selected_features = len(selected_features_indices)
    total_features_in_optimized_space = len(best_individual) 


    print("\n--- Evaluación del Mejor Subconjunto de Características ---")
    print(f"Características seleccionadas: {num_selected_features}/{total_features_in_optimized_space} (del espacio pre-procesado)")

    if num_selected_features == 0:
        print("No se seleccionó ninguna característica. No se puede evaluar el modelo final.")
        return

    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    final_classifier = RandomForestClassifier(n_estimators=500,
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

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:

        print("Advertencia: Matriz de confusión no es 2x2. Recalculando métricas.")
        TP = ((y_test == 1) & (y_pred == 1)).sum()
        TN = ((y_test == 0) & (y_pred == 0)).sum()
        FP = ((y_test == 0) & (y_pred == 1)).sum()
        FN = ((y_test == 1) & (y_pred == 0)).sum()


    dr = recall_score(y_test, y_pred, average='binary') # DR (Detection Rate) = Recall
    pr = precision_score(y_test, y_pred, average='binary') # PR (Precision)
    f1 = f1_score(y_test, y_pred, average='binary')     # F1-Score


    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    print("\nMétricas de rendimiento finales:")
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Detection Rate (DR): {dr:.4f}")
    print(f"Precision (PR): {pr:.4f}")
    print(f"F1-Score (F1): {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Número de características seleccionadas: {num_selected_features}")
    print(f"Índices de las características seleccionadas (en el espacio pre-procesado): {selected_features_indices.tolist()}")


if __name__ == "__main__":

    train_file = './UNSW_NB15_training-set.csv' 
    test_file = './UNSW_NB15_testing-set.csv'   


    import os
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"ERROR: Archivos de dataset no encontrados en '{train_file}' y '{test_file}'.")
        print("Por favor, descarga los datasets UNSW-NB15 (training-set.csv y testing-set.csv)")
        print("y actualiza las rutas de los archivos en el código.")
    else:

        X_train, X_test, y_train, y_test, original_features, categorical_features = load_and_preprocess_data(train_file, test_file)

        total_features_count = X_train.shape[1]


        bde_pop_size = 30
        bde_num_generations = 100 
        bde_F = 0.8 
        bde_CR = 0.9 


        best_features_binary_vector, final_best_fitness, fitness_history = run_bde(
            X_train, X_test, y_train, y_test,
            bde_pop_size, bde_num_generations, bde_F, bde_CR,
            total_features_count
        )

        evaluate_final_model(best_features_binary_vector, X_train, X_test, y_train, y_test, original_features, categorical_features)

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(fitness_history)
            plt.xlabel('Generación')
            plt.ylabel('Mejor Fitness (Accuracy Penalizada)')
            plt.title('Convergencia de BDE')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("\nInstala matplotlib para ver el gráfico de convergencia: pip install matplotlib")