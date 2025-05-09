# -*- coding: utf-8 -*-
# --- START OF FILE item2.py ---

import numpy as np
import os
import sys
import random
from typing import List, Tuple, Dict, Optional, Any
import time
import copy
import matplotlib.pyplot as plt

# --- Type Aliases for Clarity ---
PlaneIdx = int
PistIdx = int
seq_pista = List[PlaneIdx]
MultiPist = Dict[PistIdx, seq_pista] # The core solution representation {runway_id: [plane_id1, plane_id2, ...]}
OverallSchedule = Dict[PlaneIdx, int] # Final landing times for all planes {plane_idx: time}

# --- File Reading Function ---
def read_file(path):
    """
    Lee el archivo de instancia con el formato observado en case*.txt:
    - Linea 1: D (numero de aviones)
    - Para cada avion i = 0 to D-1:
        - Linea: E P L Ci Ck (separados por espacio)
        - Linea(s) siguientes: tau[i][0] tau[i][1] ... tau[i][D-1] (separados por espacio, pueden ocupar varias lineas)
    Retorna: num_planes, E, P, L, Ci, Ck, tau (como np.arrays)
    """
    try:
        with open(path, 'r') as file:
            # Leer lineas no vacias y quitar espacios al inicio/final
            lines = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {path}")
        sys.exit(1)

    line_idx = 0
    try:
        # Linea 1: Numero de aviones
        if line_idx >= len(lines): raise ValueError("Archivo vacío o solo espacios.")
        num_planes = int(lines[line_idx])
        line_idx += 1

        E, P, L, Ci, Ck = [], [], [], [], []
        tau_rows = [] # Lista para guardar las filas de la matriz tau

        for i in range(num_planes):
            # Leer datos del avion (E P L Ci Ck)
            if line_idx >= len(lines): raise ValueError(f"Fin de archivo inesperado al leer datos del avión {i}.")
            parts = lines[line_idx].split()
            line_idx += 1 # Avanzar a la siguiente linea (inicio de tiempos de separación)
            if len(parts) != 5:
                raise ValueError(f"Error en línea de datos del avión {i}: Se esperaban 5 valores, se obtuvieron {len(parts)} -> '{' '.join(parts)}'")
            try:
                E.append(int(parts[0]))
                P.append(int(parts[1]))
                L.append(int(parts[2]))
                Ci.append(float(parts[3]))
                Ck.append(float(parts[4]))
            except ValueError as e:
                 raise ValueError(f"Error al convertir valor en datos del avión {i}: {e} -> '{' '.join(parts)}'")

            # Leer fila de tiempos de separación (puede tomar varias lineas)
            current_sep_row = []
            while len(current_sep_row) < num_planes:
                if line_idx >= len(lines):
                    raise ValueError(f"Fin de archivo inesperado al leer tiempos de separación para avión {i} (se esperaban {num_planes}, se obtuvieron {len(current_sep_row)}).")
                
                # Leer valores de la linea actual
                sep_parts = lines[line_idx].split()
                line_idx += 1 # Asegurarse de avanzar a la siguiente linea para la proxima lectura
                try:
                    current_sep_row.extend([int(part) for part in sep_parts]) # Usar int según los casos de ejemplo
                except ValueError as e:
                    raise ValueError(f"Error al convertir valor en línea de separación {line_idx}: {e} -> '{' '.join(sep_parts)}'")
            
            # Verificar que se leyeron exactamente num_planes valores para la fila
            if len(current_sep_row) != num_planes:
                 # Este caso podría ocurrir si una línea contenía más valores de los necesarios.
                 raise ValueError(f"Error al leer tiempos de separación para avión {i}: Se esperaban {num_planes} valores, pero se procesaron {len(current_sep_row)}.")
            
            # Añadir la fila completa a la lista de filas tau
            tau_rows.append(current_sep_row)

        # Verificación final: ¿Se leyeron datos para todos los aviones?
        if len(E) != num_planes or len(tau_rows) != num_planes:
             raise ValueError("Discrepancia entre el número de aviones declarado y los datos leídos.")

    except (ValueError, IndexError) as e:
        print(f"Error al leer formato del archivo: {e}")
        sys.exit(1)
    except Exception as e: # Capturar cualquier otro error inesperado
        print(f"Error inesperado durante la lectura del archivo: {e}")
        sys.exit(1)

    # Convertir a NumPy arrays
    E = np.array(E, dtype=int)
    P_array = np.array(P, dtype=int) # Renombrar P para evitar conflicto con módulo P si existiera
    L = np.array(L, dtype=int)
    Ci = np.array(Ci, dtype=float)
    Ck = np.array(Ck, dtype=float)
    tau = np.array(tau_rows, dtype=int) # Usar int según los casos

    # --- Validaciones adicionales ---
    if tau.shape != (num_planes, num_planes):
        print(f"Advertencia: La forma de la matriz de separación es {tau.shape}, se esperaba ({num_planes}, {num_planes}).")

    if not (np.all(E <= P_array) and np.all(P_array <= L)):
        violations = [f"Avión {k+1}: E={E[k]}, P={P_array[k]}, L={L[k]}" for k in range(num_planes) if not (E[k] <= P_array[k] <= L[k])]
        print(f"Advertencia: No todos los aviones cumplen E <= P <= L. Violaciones:\n" + "\n".join(violations))
        
    diag_tau = np.diag(tau)
    if not np.all(diag_tau >= 99999):
         print(f"Advertencia: No todos los elementos diagonales de tau son >= 99999. Encontrados: {diag_tau[diag_tau < 99999]}")

    return num_planes, E, P_array, L, Ci, Ck, tau

# --- Cost Calculation Function ---
def calculate_total_cost(overall_schedule: OverallSchedule, P_array: np.ndarray, Ci: np.ndarray, Ck: np.ndarray) -> float:
    """Calcula el costo total de penalización para un horario completo."""
    if not overall_schedule: return float('inf')
    total_cost = 0.0
    num_planes_total = len(P_array)
    scheduled_planes = set(overall_schedule.keys())

    # Verificar si faltan aviones en el horario
    if len(scheduled_planes) != num_planes_total:
        # print(f"Cost Warning: Schedule incomplete. Expected {num_planes_total}, got {len(scheduled_planes)}")
        return float('inf') # Un horario incompleto no es válido para el costo final

    for plane_idx, landing_time in overall_schedule.items():
        if not (0 <= plane_idx < num_planes_total):
            print(f"Cost Error: Índice de avión inválido {plane_idx}")
            return float('inf')
        penalty = Ci[plane_idx] * max(0, P_array[plane_idx] - landing_time) + \
                  Ck[plane_idx] * max(0, landing_time - P_array[plane_idx])
        total_cost += penalty

    if np.isnan(total_cost) or np.isinf(total_cost):
        print("Cost Warning: Cálculo de costo resultó en NaN o Inf.")
        return float('inf')

    return total_cost

# --- Schedule Calculation for Multi-Runway Layout ---
def get_schedule_from_multi_runway_layout(
    multi_runway_layout: MultiPist,
    num_pist: int,
    E: np.ndarray, L: np.ndarray, tau: np.ndarray
) -> Tuple[Optional[OverallSchedule], bool]:
    """
    Calcula los tiempos de aterrizaje para una disposición dada de múltiples pistas.
    Comprueba restricciones E, L y tau dentro de cada pista.
    Devuelve: OverallSchedule {plane_idx: tiempo} o None si no es factible, junto con un indicador de factibilidad.
    """
    overall_schedule: OverallSchedule = {}
    # Estado por pista
    last_landing_time_on_runway: Dict[PistIdx, int] = {r: -1 for r in range(num_pist)}
    last_plane_on_runway: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_pist)}
    
    processed_planes = set() # Para verificar si se procesan todos

    for runway_idx in range(num_pist):
        # Si la pista no existe en el layout (no debería pasar si está bien formado), tratarla como vacía
        if runway_idx not in multi_runway_layout:
            multi_runway_layout[runway_idx] = []

        # Procesar la secuencia de esta pista
        for plane_idx in multi_runway_layout[runway_idx]:
            # Verificar duplicados (un avión no puede estar en dos sitios)
            if plane_idx in processed_planes:
                 print(f"Schedule Error: Avión {plane_idx} aparece múltiples veces en el layout.")
                 return None, False
            processed_planes.add(plane_idx)

            # Verificar índice de avión válido
            if not (0 <= plane_idx < len(E)):
                print(f"Schedule Error: Índice de avión inválido {plane_idx} encontrado en layout.")
                return None, False

            # Calcular tiempo mínimo de inicio
            min_start_time = E[plane_idx]
            earliest_after_separation = 0

            # Calcular restricción de separación si no es el primer avión en la pista
            prev_plane = last_plane_on_runway[runway_idx]
            if prev_plane != -1:
                # Verificar índices para tau
                if not (0 <= prev_plane < tau.shape[0] and 0 <= plane_idx < tau.shape[1]):
                     print(f"Schedule Error: Índices inválidos para tau[{prev_plane}][{plane_idx}].")
                     return None, False
                separation_needed = tau[prev_plane][plane_idx]
                # Verificar separación infactible
                if separation_needed >= 99999:
                    # print(f"Schedule Info: Separación infactible tau[{prev_plane}][{plane_idx}] en pista {runway_idx}") # Verbose
                    return None, False
                earliest_after_separation = last_landing_time_on_runway[runway_idx] + separation_needed

            # Tiempo de aterrizaje real es el máximo de Ek y el tiempo post-separación
            actual_landing_time = max(min_start_time, earliest_after_separation)

            # Verificar contra Lk
            if actual_landing_time > L[plane_idx]:
                # print(f"Schedule Info: Violación Lk para avión {plane_idx} en pista {runway_idx}. Req: {actual_landing_time}, Max: {L[plane_idx]}") # Verbose
                return None, False

            # Asignar tiempo y actualizar estado de la pista
            overall_schedule[plane_idx] = actual_landing_time
            last_landing_time_on_runway[runway_idx] = actual_landing_time
            last_plane_on_runway[runway_idx] = plane_idx

    # Comprobación final: ¿Están todos los aviones del problema en el horario?
    if len(processed_planes) != len(E):
        # print(f"Schedule Warning: Layout incompleto. Esperados: {len(E)}, Procesados: {len(processed_planes)}")
        # Consideramos un layout incompleto como infactible para evaluación final
        return None, False

    return overall_schedule, True

# --- Full Evaluation of Multi-Runway Layout ---
def evaluate_multi_runway_layout(
    multi_runway_layout: MultiPist, num_pist: int,
    E: np.ndarray, P_array: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray
) -> Tuple[float, Optional[OverallSchedule], bool]:
    """
    Evalúa un layout multi-pista: calcula horario, verifica factibilidad completa, calcula costo.
    """
    # 1. Calcular el horario y verificar factibilidad básica (E, L, tau)
    overall_schedule, feasible = get_schedule_from_multi_runway_layout(multi_runway_layout, num_pist, E, L, tau)

    # Si las restricciones básicas fallan o el horario está incompleto, es infactible
    if not feasible or overall_schedule is None:
        return float('inf'), None, False

    # 2. Verificar que el horario resultante contiene TODOS los aviones
    if len(overall_schedule) != len(E): # len(E) es num_planes
        # print(f"Evaluate Error: Horario incompleto. Esperados {len(E)}, Obtenidos {len(overall_schedule)}")
        return float('inf'), None, False # Un horario incompleto es infactible

    # 3. Calcular el costo total si es factible y completo
    cost = calculate_total_cost(overall_schedule, P_array, Ci, Ck)
    return cost, overall_schedule, True

# --- Stochastic Greedy Algorithm (Multi-Runway) ---
def greedy_stochastic_multi(
    num_planes: int, num_pist: int, E: np.ndarray, P_array: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, rcl_size: int = 3, seed: Optional[int] = None
    ) -> Tuple[Optional[MultiPist], float, bool]:
    """
    Algoritmo Greedy Estocástico Multi-Pista.
    Construye un layout asignando aviones uno a uno basados en una RCL de (avión, pista, tiempo).
    Devuelve el layout, su costo *recalculado* final, y factibilidad.
    """
    if seed is not None: random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    # Estructura de la solución que se construye
    solution_layout: MultiPist = {r: [] for r in range(num_pist)}
    # Estado de las pistas para la construcción
    last_landing_time_on_runway: Dict[PistIdx, int] = {r: -1 for r in range(num_pist)}
    last_plane_on_runway: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_pist)}
    
    planes_scheduled_count = 0
    construction_feasible = True

    while unscheduled_planes:
        candidates = [] # Lista de diccionarios: {'plane': k, 'runway': r, 'time': t}
        min_cand_time = float('inf')

        for k_plane in unscheduled_planes:
            for r_idx in range(num_pist):
                min_start_time = E[k_plane]
                earliest_after_sep = 0
                prev_p = last_plane_on_runway[r_idx]

                if prev_p != -1:
                    # Índices válidos ya verificados en evaluate, pero buena práctica aquí también?
                    if not (0 <= prev_p < tau.shape[0] and 0 <= k_plane < tau.shape[1]):
                         print(f"Greedy Sto Error: Índices inválidos tau[{prev_p}][{k_plane}]")
                         construction_feasible = False; break # Error crítico
                    sep = tau[prev_p][k_plane]
                    if sep >= 99999: continue # Opción no factible en esta pista
                    earliest_after_sep = last_landing_time_on_runway[r_idx] + sep
                
                possible_time = max(min_start_time, earliest_after_sep)
                
                if possible_time <= L[k_plane]: # Si cumple Lk
                    candidates.append({'plane': k_plane, 'runway': r_idx, 'time': possible_time})
                    min_cand_time = min(min_cand_time, possible_time) # Para la lógica de RCL si es necesaria
            if not construction_feasible: break # Salir del bucle de aviones si hay error
        if not construction_feasible: break # Salir del bucle while si hay error

        if not candidates:
            # No hay movimientos posibles para los aviones restantes
            # print(f"Greedy Sto Info (Seed {seed}): No hay candidatos factibles. No programados: {unscheduled_planes}")
            construction_feasible = False
            break

        # Ordenar candidatos por tiempo (menor es mejor) y crear RCL
        candidates.sort(key=lambda x: x['time'])
        current_rcl_limit = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_limit]
        
        # Elegir aleatoriamente de la RCL
        chosen_candidate = random.choice(rcl)
        chosen_plane = chosen_candidate['plane']
        chosen_runway = chosen_candidate['runway']
        chosen_time = chosen_candidate['time']

        # Asignar y actualizar estado
        solution_layout[chosen_runway].append(chosen_plane)
        last_landing_time_on_runway[chosen_runway] = chosen_time
        last_plane_on_runway[chosen_runway] = chosen_plane
        unscheduled_planes.remove(chosen_plane)
        planes_scheduled_count += 1

    # Verificar si la construcción fue factible y completa
    if not construction_feasible or planes_scheduled_count != num_planes:
        # print(f"Greedy Sto Info (Seed {seed}): Construcción fallida o incompleta ({planes_scheduled_count}/{num_planes}).")
        return None, float('inf'), False

    # Recalcular costo final y factibilidad usando la función de evaluación estándar
    # Esto asegura consistencia con cómo se evalúan las soluciones en HC y GRASP
    final_cost, _, final_feasible = evaluate_multi_runway_layout(solution_layout, num_pist, E, P_array, L, Ci, Ck, tau)

    if not final_feasible:
        # print(f"Greedy Sto Warning (Seed {seed}): Layout construido es INFACTIBLE en la re-evaluación final.")
        return None, float('inf'), False

    return solution_layout, final_cost, True

# --- Hill Climbing (Multi-Runway) ---
def hill_climbing_multi(
    initial_layout: MultiPist, num_pist: int,
    E: np.ndarray, P_array: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    use_best_improvement: bool) -> Tuple[MultiPist, float]:
    """
    Búsqueda Local Hill Climbing para múltiples pistas.
    Operadores: Intercambio Intra-Pista, Movimiento Inter-Pista (al final).
    Devuelve el layout localmente óptimo y su costo.
    """
    current_layout = copy.deepcopy(initial_layout)
    # Evaluar la solución inicial para obtener costo y verificar factibilidad
    current_cost, _, feasible = evaluate_multi_runway_layout(current_layout, num_pist, E, P_array, L, Ci, Ck, tau)

    # Si la solución inicial no es factible, no se puede mejorar
    if not feasible:
        # print("HC Warning: Layout inicial es infactible.")
        return initial_layout, float('inf')

    while True:
        best_neighbor_layout = None # Guarda el mejor vecino encontrado en esta iteración (para BI)
        best_neighbor_cost = current_cost
        moved_in_iter = False # Indica si se realizó algún movimiento (para FI) o si se encontró mejora (para BI)

        # --- 1. Operador: Intercambio Intra-Pista ---
        for r_idx in range(num_pist):
            pista_seq = current_layout[r_idx]
            if len(pista_seq) < 2: continue # Necesita al menos 2 aviones para intercambiar

            for i in range(len(pista_seq)):
                for j in range(i + 1, len(pista_seq)):
                    # Crear vecino haciendo una copia y el intercambio
                    neighbor_layout = copy.deepcopy(current_layout)
                    neighbor_layout[r_idx][i], neighbor_layout[r_idx][j] = neighbor_layout[r_idx][j], neighbor_layout[r_idx][i]

                    # Evaluar el vecino
                    cost_n, _, feas_n = evaluate_multi_runway_layout(neighbor_layout, num_pist, E, P_array, L, Ci, Ck, tau)

                    # Si es factible y mejora el *mejor costo encontrado hasta ahora* en esta iteración BI
                    if feas_n and cost_n < best_neighbor_cost:
                        if use_best_improvement:
                            best_neighbor_cost = cost_n
                            best_neighbor_layout = neighbor_layout # Guardar el mejor vecino
                            moved_in_iter = True # Indicar que *es posible* mejorar
                        else: # Primera Mejora (FI)
                            # Moverse inmediatamente al primer vecino que mejore
                            current_layout = neighbor_layout
                            current_cost = cost_n
                            moved_in_iter = True
                            # print(f"  HC FI (IntraSwap): Moved to {current_cost:.2f}") # Verbose
                            break # Salir del bucle j
                if not use_best_improvement and moved_in_iter: break # Salir del bucle i
            if not use_best_improvement and moved_in_iter: break # Salir del bucle r_idx
        # Si se hizo un movimiento en FI, reiniciar el bucle while desde la nueva solución
        if not use_best_improvement and moved_in_iter:
            continue # Va al inicio del while True

        # --- 2. Operador: Movimiento Inter-Pista (Mover al final) ---
        if num_pist > 1: # Solo tiene sentido si hay más de 1 pista
            for src_r_idx in range(num_pist):
                # Iterar sobre posiciones para evitar problemas si se modifica la lista
                for plane_pos_in_src in range(len(current_layout[src_r_idx])):
                     # Verificar que el índice sigue siendo válido (podría cambiar si se modificó antes en FI?) - Mejor usar copia siempre
                    if plane_pos_in_src >= len(current_layout[src_r_idx]): continue # Índice ya no válido

                    for dest_r_idx in range(num_pist):
                        if src_r_idx == dest_r_idx: continue # Mover a una pista diferente

                        # Crear vecino
                        neighbor_layout = copy.deepcopy(current_layout)
                        # Sacar el avión de la pista origen
                        try:
                            moved_plane = neighbor_layout[src_r_idx].pop(plane_pos_in_src)
                        except IndexError:
                             print(f"HC Error: Índice {plane_pos_in_src} fuera de rango al mover de pista {src_r_idx} (len={len(neighbor_layout[src_r_idx])}).")
                             continue # Saltar este movimiento problemático
                        # Añadir al final de la pista destino
                        neighbor_layout[dest_r_idx].append(moved_plane)

                        # Evaluar
                        cost_n, _, feas_n = evaluate_multi_runway_layout(neighbor_layout, num_pist, E, P_array, L, Ci, Ck, tau)

                        if feas_n and cost_n < best_neighbor_cost:
                            if use_best_improvement:
                                best_neighbor_cost = cost_n
                                best_neighbor_layout = neighbor_layout
                                moved_in_iter = True # Indicar que se encontró mejora
                            else: # Primera Mejora (FI)
                                current_layout = neighbor_layout
                                current_cost = cost_n
                                moved_in_iter = True
                                # print(f"  HC FI (InterMove): Moved to {current_cost:.2f}") # Verbose
                                break # Salir bucle dest_r_idx
                    if not use_best_improvement and moved_in_iter: break # Salir bucle plane_pos_in_src
                if not use_best_improvement and moved_in_iter: break # Salir bucle src_r_idx
            # Si se hizo un movimiento en FI, reiniciar el bucle while
            if not use_best_improvement and moved_in_iter:
                continue

        # --- Decisión al final de la iteración del bucle While ---
        if use_best_improvement:
            # Si se encontró un mejor vecino en toda la exploración (BI)
            if moved_in_iter and best_neighbor_layout is not None: # moved_in_iter es True si best_neighbor_cost < current_cost
                # print(f"  HC BI: Moving to best neighbor found. Cost: {best_neighbor_cost:.2f}") # Verbose
                current_layout = best_neighbor_layout
                current_cost = best_neighbor_cost
                # Continuar el bucle while para buscar mejoras desde la nueva solución
            else:
                # No se encontró ninguna mejora en toda la vecindad
                # print(f"  HC BI: Local optimum reached. Cost: {current_cost:.2f}") # Verbose
                break # Salir del bucle while
        else:
            # Para FI, si moved_in_iter es False aquí, significa que no se encontró
            # ninguna mejora en toda la vecindad o ya se hizo el movimiento y se reinició el while.
            # Si no hubo movimiento, salir.
            if not moved_in_iter:
                # print(f"  HC FI: Local optimum reached. Cost: {current_cost:.2f}") # Verbose
                break # Salir del bucle while

    # Devolver la mejor solución encontrada (óptimo local)
    return current_layout, current_cost

# --- Iterated Greedy with Hill Climbing ---
def grasp_hc(
    num_planes: int, num_pist: int, E: np.ndarray, P_array: np.ndarray,
    L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    num_restarts: int, # Número de veces que se aplica el restart (llamada a greedy + HC)
    rcl_size_greedy: int,
    use_mejor_mejora_hc: bool,
    initial_deterministic_layout: Optional[MultiPist] # Layout del greedy determinista
    ) -> Tuple[Optional[MultiPist], float]:
    """
    Implementa el algoritmo Iterated Greedy con Búsqueda Local (Hill Climbing).
    En cada restart, genera una solución con Greedy Estocástico y la mejora con HC.
    """
    best_overall_layout: Optional[MultiPist] = None
    best_overall_cost: float = float('inf')
    start_time_ig = time.time()

    hc_type_str = 'BI (Mejor Mejora)' if use_mejor_mejora_hc else 'FI (Alguna Mejora)'
    print(f"\n--- Iniciando Iterated Greedy con HC {hc_type_str} ---")
    print(f"Número de Restarts (Greedy Estocástico + HC): {num_restarts}")
    print(f"RCL Size para Greedy Estocástico: {rcl_size_greedy}")

    # Opcional: Mejorar la solución determinista una vez al inicio como referencia/baseline
    if initial_deterministic_layout:
        print("[IG] Mejorando solución inicial determinista (una vez)...")
        # Evaluar primero si el layout determinista es completo
        planes_in_det = sum(len(seq) for seq in initial_deterministic_layout.values())
        if planes_in_det == num_planes:
            cost_det_check, _, feas_det_check = evaluate_multi_runway_layout(
                initial_deterministic_layout, num_pist, E, P_array, L, Ci, Ck, tau
            )
            if feas_det_check:
                det_hc_layout, det_hc_cost = hill_climbing_multi(
                    initial_deterministic_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora_hc
                )
                if det_hc_cost < best_overall_cost:
                    best_overall_cost = det_hc_cost
                    best_overall_layout = det_hc_layout
                    print(f"  Costo después de HC en Determinista: {best_overall_cost:.2f}")
            else:
                 print("  Layout determinista inicial es infactible, no se puede mejorar.")
        else:
             print(f"  Layout determinista inicial incompleto ({planes_in_det}/{num_planes}), no se puede mejorar.")

    # Bucle principal de Iterated Greedy (restarts)
    for i in range(num_restarts):
        print(f"\n[IG] Restart {i+1}/{num_restarts}")

        # 1. Generar Solución Base con Greedy Estocástico
        current_seed = i # Usar i como semilla para reproducibilidad simple
        current_seed = random.randint(0, 100000) # O usar semilla aleatoria
        print(f"  Generando solución base con Greedy Estocástico (seed={current_seed})...") # Verbose
        base_layout, base_cost_reported, base_feasible = greedy_stochastic_multi(
            num_planes, num_pist, E, P_array, L, Ci, Ck, tau,
            rcl_size=rcl_size_greedy, seed=current_seed
        )

        if not base_feasible or base_layout is None:
            print(f"  Greedy Estocástico (seed={current_seed}) no encontró solución factible. Saltando restart.")
            continue

        # Doble verificación: que el layout devuelto sea completo y factible al reevaluar
        base_cost_eval, _, base_feasible_eval = evaluate_multi_runway_layout(
            base_layout, num_pist, E, P_array, L, Ci, Ck, tau
        )
        if not base_feasible_eval:
             print(f"  Error: Layout de Greedy Estocástico (seed={current_seed}) infactible al reevaluar. Saltando restart.")
             continue
        # Usar el costo reevaluado por consistencia
        base_cost = base_cost_eval

        print(f"  Solución base generada (Costo: {base_cost:.2f}). Ejecutando Hill Climbing...")

        # 2. Búsqueda Local (Hill Climbing)
        local_opt_layout, local_opt_cost = hill_climbing_multi(
            base_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora_hc
        )

        print(f"  Óptimo local encontrado en este restart. Costo: {local_opt_cost:.2f}")

        # 3. Actualizar Mejor Solución Global
        if local_opt_cost < best_overall_cost:
            best_overall_cost = local_opt_cost
            best_overall_layout = local_opt_layout
            print(f"    >> ¡Nuevo mejor costo global encontrado!: {best_overall_cost:.2f} (en restart {i+1})")

    end_time_ig = time.time()
    print(f"\n[IG] Proceso Iterated Greedy ({hc_type_str}) completado en {end_time_ig - start_time_ig:.4f} segundos.")

    if best_overall_layout is None:
        print("[IG] No se encontró ninguna solución factible en todo el proceso.")
        return None, float('inf')

    # Verificación final de la mejor solución encontrada
    final_cost, _, final_feasible = evaluate_multi_runway_layout(best_overall_layout, num_pist, E, P_array, L, Ci, Ck, tau)

    if not final_feasible:
        print(f"[IG] ERROR CRÍTICO: La mejor solución guardada (costo rastreado: {best_overall_cost:.2f}) resultó infactible en la verificación final.")
        # Esto no debería ocurrir si la lógica es correcta
        return None, float('inf')

    # Corregir si hay discrepancia mínima por flotantes o si HC no actualizó bien
    if abs(final_cost - best_overall_cost) > 1e-5:
        print(f"[IG] Advertencia de Verificación Final: Costo verificado ({final_cost:.2f}) difiere del rastreado ({best_overall_cost:.2f}). Usando el costo verificado.")
        best_overall_cost = final_cost

    print(f"[IG] Mejor costo final verificado: {best_overall_cost:.2f}")
    return best_overall_layout, best_overall_cost

# --- UTILITY: Print solution details ---
def print_solution_multi(
    overall_schedule: Optional[OverallSchedule],
    total_cost: float,
    feasible: bool,
    multi_runway_layout: Optional[MultiPist] = None,
    num_pist: Optional[int] = None
):
    """Imprime los detalles de la solución multi-pista."""
    cost_str = "inf" if total_cost == float('inf') else f"{total_cost:.2f}"
    if not feasible or overall_schedule is None:
        print(f"No se encontró solución factible. Costo: {cost_str}")
        if multi_runway_layout:
            print("Layout (parcial/infactible):")
            # Imprimir layout incluso si es infactible para depuración
            for r in range(num_pist if num_pist is not None else len(multi_runway_layout)):
                 planes_str = str(multi_runway_layout.get(r, [])) # Usar get con default
                 print(f"  Pista {r+1}: {planes_str}")
        return

    print(f"Costo total: {total_cost:.2f}")

    # Imprimir asignación detallada por pista si se proporciona el layout
    if multi_runway_layout and num_pist is not None:
        print("\nAsignación y Tiempos por Pista:")
        planes_on_runway_details = {r: [] for r in range(num_pist)}

        # Crear mapa de avión a pista para facilitar la asignación
        plane_to_runway_map: Dict[PlaneIdx, PistIdx] = {}
        if multi_runway_layout: # Asegurarse de que no es None
            for r_idx, planes_in_r in multi_runway_layout.items():
                if r_idx >= num_pist: continue # Ignorar pistas fuera de rango si existen
                for p_idx in planes_in_r:
                    plane_to_runway_map[p_idx] = r_idx

        # Ordenar aviones por tiempo de aterrizaje para impresión
        sorted_planes_by_time = sorted(overall_schedule.items(), key=lambda item: item[1])

        # Llenar los detalles por pista
        for p_idx, landing_time in sorted_planes_by_time:
            if p_idx in plane_to_runway_map:
                 r_assigned = plane_to_runway_map[p_idx]
                 # Formato: Avión Indice+1 @ Tiempo
                 planes_on_runway_details[r_assigned].append(f"Avión {p_idx+1}@{landing_time}")
            else:
                 # Esto indica una inconsistencia entre el schedule y el layout
                 print(f"Advertencia: Avión {p_idx+1} está en horario pero no mapeado a una pista del layout.")

        # Imprimir detalles por pista
        for r_idx in range(num_pist):
            details_str = ", ".join(planes_on_runway_details[r_idx])
            print(f"  Pista {r_idx+1}: [{details_str}]")

    # Imprimir horario general ordenado por tiempo
    print("\nHorario General Ordenado (Avión: Tiempo):")
    print("Avión\tTiempo de aterrizaje")
    print("-" * 25)
    # Ordenar el horario general por tiempo de aterrizaje
    sorted_schedule_items = sorted(overall_schedule.items(), key=lambda item: item[1])
    for plane_idx, landing_time in sorted_schedule_items:
        print(f"{plane_idx + 1}\t{landing_time}") # Mostrar índice base 1

# --- Main Execution Logic ---
def main():
    DEFAULT_CASE_DIR = "casos" # Directorio relativo donde están los casos

    # --- Selección de Caso ---
    while True:
        select_str = input("Selecciona el caso (1-4) o 0 para salir: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n")
        try:
            select = int(select_str)
            if select == 0: print("Saliendo..."); sys.exit()
            if 1 <= select <= 4: break
            else: print("Opción no válida. Por favor, elija entre 1 y 4.")
        except ValueError:
            print("Entrada inválida. Por favor ingrese un número.")

    # --- Selección de Número de Pistas ---
    while True:
        try:
            num_runways_str = input("Ingrese el número de pistas a utilizar (1 o 2): ")
            num_pists = int(num_runways_str)
            if num_pists in [1, 2]: break
            else: print("Número de pistas debe ser 1 o 2.")
        except ValueError:
            print("Por favor ingrese un número entero (1 o 2).")

    # --- Construcción de Ruta al Archivo ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Usar directorio del script o directorio actual como base
    base_dir = script_dir if script_dir else os.getcwd()
    cases_dir_path = os.path.join(base_dir, DEFAULT_CASE_DIR)
    filename = os.path.join(cases_dir_path, f'case{select}.txt')
    print(f"\nLeyendo instancia desde: {filename}")

    try:
        # --- Lectura de Datos ---
        num_planes, E, P_array, L, Ci, Ck, tau = read_file(filename)
        print("-" * 40 + f"\nLectura exitosa. Aviones: {num_planes}, Pistas: {num_pists}" + "\n" + "-" * 40)

        # --- Ejecución Greedy Determinista (Baseline) ---
        # Necesario para la comparación y opcionalmente para iniciar IG
        print("\n--- Greedy Determinista (Multi-Pista) ---")
        start_det = time.time()
        det_layout, det_cost_reported, det_feasible = greedy_stochastic_multi(
             num_planes, num_pists, E, P_array, L, Ci, Ck, tau
        )
        det_time = time.time() - start_det
        print(f"Tiempo ejecución determinista: {det_time:.4f}s")

        # Verificar y obtener horario/costo final del determinista
        det_schedule, det_cost_final = None, float('inf')
        if det_feasible and det_layout:
            cost_chk, sched_chk, feas_chk = evaluate_multi_runway_layout(
                 det_layout, num_pists, E, P_array, L, Ci, Ck, tau
            )
            if feas_chk and sched_chk is not None:
                 det_schedule = sched_chk
                 det_cost_final = cost_chk
                 det_feasible = True # Confirmar factibilidad
            else:
                 det_feasible = False # Marcar como infactible si la re-evaluación falla
                 det_cost_final = float('inf')
                 det_layout = None # Invalidar layout
        else: # Si el greedy ya dijo que era infactible
             det_cost_final = float('inf')
             det_feasible = False
             det_layout = None

        print("Resultado Greedy Determinista:")
        print_solution_multi(det_schedule, det_cost_final, det_feasible, det_layout, num_pists)

        # --- Parámetros para Iterated Greedy ---
        try:
             num_restarts_str = input(f"Ingrese el número de restarts para Iterated Greedy (ej: 10, 50): ")
             NUM_RESTARTS = int(num_restarts_str)
             if NUM_RESTARTS <= 0: NUM_RESTARTS = 10; print("Número inválido, usando 10 restarts.")
        except ValueError:
             NUM_RESTARTS = 10; print("Entrada inválida, usando 10 restarts por defecto.")

        RCL_SIZE_GREEDY = 3 # Puedes hacerlo configurable si quieres
        print(f"\nConfigurando Iterated Greedy: Restarts={NUM_RESTARTS}, RCL Size Greedy={RCL_SIZE_GREEDY}")

        # --- Ejecución del Iterated Greedy con HC ---
        print("\n" + "=" * 40 + "\nALGORITMO ITERATED GREEDY con HILL CLIMBING\n" + "=" * 40)

        plot_labels_ig = []
        plot_costs_ig = []
        plot_times_ig = []

        # 1. Ejecutar con HC Primera Mejora (Alguna-Mejora)
        start_ig_fi = time.time()
        ig_fi_layout, ig_fi_cost = grasp_hc(
            num_planes, num_pists, E, P_array, L, Ci, Ck, tau,
            num_restarts=NUM_RESTARTS,
            rcl_size_greedy=RCL_SIZE_GREEDY,
            use_mejor_mejora_hc=False, # Primera Mejora
            initial_deterministic_layout=det_layout # Pasa el layout determinista verificado
        )
        time_ig_fi = time.time() - start_ig_fi

        # Evaluar resultado final para mostrar e graficar
        ig_fi_schedule, ig_fi_feasible = (None, False)
        if ig_fi_layout and ig_fi_cost != float('inf'):
             # El costo devuelto por IG ya está verificado, solo necesitamos el schedule
             _, ig_fi_schedule, ig_fi_feasible = evaluate_multi_runway_layout(ig_fi_layout, num_pists, E, P_array, L, Ci, Ck, tau)
             # Si por alguna razón la reevaluación aquí falla, marcar como infactible
             if not ig_fi_feasible: ig_fi_cost = float('inf')

        print("\nMejor solución encontrada (Iterated Greedy + HC Primera Mejora):")
        print_solution_multi(ig_fi_schedule, ig_fi_cost, ig_fi_feasible, ig_fi_layout, num_pists)

        if ig_fi_feasible:
            plot_labels_ig.append("IG + HC Primera Mejora")
            plot_costs_ig.append(ig_fi_cost)
            plot_times_ig.append(time_ig_fi)

        # 2. Ejecutar con HC Mejor Mejora
        start_ig_bi = time.time()
        ig_bi_layout, ig_bi_cost = grasp_hc(
            num_planes, num_pists, E, P_array, L, Ci, Ck, tau,
            num_restarts=NUM_RESTARTS,
            rcl_size_greedy=RCL_SIZE_GREEDY,
            use_mejor_mejora_hc=True, # Mejor Mejora
            initial_deterministic_layout=det_layout # Pasa el layout determinista verificado
        )
        time_ig_bi = time.time() - start_ig_bi

        # Evaluar resultado final para mostrar e graficar
        ig_bi_schedule, ig_bi_feasible = (None, False)
        if ig_bi_layout and ig_bi_cost != float('inf'):
             _, ig_bi_schedule, ig_bi_feasible = evaluate_multi_runway_layout(ig_bi_layout, num_pists, E, P_array, L, Ci, Ck, tau)
             if not ig_bi_feasible: ig_bi_cost = float('inf')

        print("\nMejor solución encontrada (Iterated Greedy + HC Mejor Mejora):")
        print_solution_multi(ig_bi_schedule, ig_bi_cost, ig_bi_feasible, ig_bi_layout, num_pists)

        if ig_bi_feasible:
            plot_labels_ig.append("IG + HC Mejor Mejora")
            plot_costs_ig.append(ig_bi_cost)
            plot_times_ig.append(time_ig_bi)

        # --- Comparación Final y Gráficos ---
        print("\n" + "=" * 40 + "\nCOMPARACIÓN FINAL ITERATED GREEDY\n" + "=" * 40)
        fi_str_ig = f"{ig_fi_cost:.2f}" if ig_fi_feasible else "inf"
        bi_str_ig = f"{ig_bi_cost:.2f}" if ig_bi_feasible else "inf"
        print(f"Costo IG + HC Primera Mejora : {fi_str_ig}")
        print(f"Costo IG + HC Mejor Mejora   : {bi_str_ig}")
        # Opcional: Comparar con el determinista si fue factible
        if det_feasible: print(f"Costo Greedy Determinista    : {det_cost_final:.2f}")
        print("=" * 40)

        # Gráfico de Costos IG
        if plot_labels_ig and plot_costs_ig:
            plt.figure(figsize=(8, 6))
            bars = plt.bar(plot_labels_ig, plot_costs_ig, color=['skyblue', 'lightcoral'])
            max_cost_plot = 0
            if plot_costs_ig: # Calcular max solo si hay costos
                 finite_costs = [c for c in plot_costs_ig if c != float('inf')]
                 if finite_costs:
                      max_cost_plot = max(finite_costs)

            for bar in bars:
                yval = bar.get_height()
                if yval != float('inf'):
                     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(1, max_cost_plot), f'{yval:.2f}', ha='center', va='bottom')
            plt.ylabel('Costo Total Final')
            plt.title(f'Comparación Costos Iterated Greedy (Restarts: {NUM_RESTARTS})\nCaso {select}, Pistas: {num_pists}')
            plt.ylim(top=max(1, max_cost_plot) * 1.1) # Ajustar límite superior eje Y
            plt.tight_layout()
            plot_filename_ig_cost = f"comparacion_ig_costo_caso{select}_pistas{num_pists}_restarts{NUM_RESTARTS}.png"
            plt.savefig(plot_filename_ig_cost); print(f"\nGráfico IG Costos guardado: {plot_filename_ig_cost}"); plt.show()
        else:
             print("\nNo se generó gráfico de costos IG (sin resultados factibles).")

        # Gráfico de Tiempos IG
        if plot_labels_ig and plot_times_ig: # Asegurarse que las listas no estén vacías
            plt.figure(figsize=(8, 6))
            bars_t = plt.bar(plot_labels_ig, plot_times_ig, color=['gold', 'lightsalmon'])
            max_time_plot = 0
            if plot_times_ig: max_time_plot = max(plot_times_ig)
            for bar in bars_t:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(1, max_time_plot), f'{yval:.2f}s', ha='center', va='bottom')
            plt.ylabel('Tiempo Ejecución (s)')
            plt.title(f'Comparación Tiempos Iterated Greedy (Restarts: {NUM_RESTARTS})\nCaso {select}, Pistas: {num_pists}')
            plt.ylim(top=max(1, max_time_plot) * 1.1) # Ajustar límite superior eje Y
            plt.tight_layout()
            plot_filename_ig_time = f"comparacion_ig_tiempo_caso{select}_pistas{num_pists}_restarts{NUM_RESTARTS}.png"
            plt.savefig(plot_filename_ig_time); print(f"\nGráfico IG Tiempos guardado: {plot_filename_ig_time}"); plt.show()
        else:
             print("\nNo se generó gráfico de tiempos IG (sin resultados factibles).")

        # El histograma del greedy estocástico ya no se genera aquí porque se llama dentro de IG.
        # Si quisieras analizarlo, tendrías que modificar IG para guardar los costos base.

    except FileNotFoundError:
        print(f"Error Crítico: No se encontró el archivo de datos {filename}")
    except (ValueError, IndexError) as e:
         print(f"Error Crítico durante la lectura o procesamiento inicial: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"Error Crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
# --- END OF FILE item2.py ---