import numpy as np
import os
import sys
import random
from typing import List, Tuple, Dict, Optional, Any
import time

L_VIOLATION_PENALTY_PER_UNIT = 100000.0  # Penalización grande por cada unidad de tiempo más allá de L[k]

def read_file(path):
    """
    Lee los datos del problema desde un archivo.
    """
    try:
        with open(path, 'r') as file:
            data = file.readlines()
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {path}")
        sys.exit(1)

    index = 0
    try:
        num_planes = int(data[index].strip())
        index += 1

        E_list, P_list, L_list_data, Ci_list, Ck_list = [], [], [], [], []
        tau_list_data = []

        for i in range(num_planes):
            values = data[index].strip().split()
            if len(values) < 5:
                raise ValueError(f"Error parseando datos del avión {i+1} línea {index+1}: Se esperaban 5 valores, se obtuvieron {len(values)}")
            E_list.append(int(values[0]))
            P_list.append(int(values[1]))
            L_list_data.append(int(values[2]))
            Ci_list.append(float(values[3]))
            Ck_list.append(float(values[4]))
            index += 1

            sep_row = []
            while len(sep_row) < num_planes:
                if index >= len(data):
                     raise ValueError(f"Error parseando tiempos de separación para el avión {i+1}: Fin de archivo inesperado.")
                line_values = data[index].strip().split()
                if not line_values: # Saltar líneas vacías si las hubiera
                    index += 1
                    continue
                try:
                    sep_row.extend(list(map(int, line_values)))
                except ValueError:
                     raise ValueError(f"Error parseando tiempos de separación en línea {index+1}: Valor no entero encontrado.")
                index += 1
            if len(sep_row) != num_planes:
                 raise ValueError(f"Error parseando tiempos de separación para el avión {i+1}: Se esperaban {num_planes} valores, se obtuvieron {len(sep_row)}")
            tau_list_data.append(sep_row)

        if len(E_list) != num_planes or len(tau_list_data) != num_planes:
             raise ValueError("Discrepancia entre el número declarado de aviones y los datos leídos.")
        for row_idx, row_val in enumerate(tau_list_data): # Comprobación adicional de la matriz tau
            if len(row_val) != num_planes:
                 raise ValueError(f"La fila {row_idx} de la matriz de separación tiene una longitud inconsistente: {len(row_val)}, se esperaba {num_planes}")


    except (ValueError, IndexError) as e:
        print(f"Error leyendo el formato del archivo: {e}")
        sys.exit(1)

    E_np = np.array(E_list)
    P_np = np.array(P_list)
    L_np = np.array(L_list_data)
    Ci_np = np.array(Ci_list)
    Ck_np = np.array(Ck_list)
    tau_np = np.array(tau_list_data)

    if not np.all(E_np <= P_np) or not np.all(P_np <= L_np):
        # Esta advertencia se basa en los datos originales.
        print("Advertencia: No todos los aviones cumplen la restricción E <= P <= L según el archivo de entrada.")

    return num_planes, E_np, P_np, L_np, Ci_np, Ck_np, tau_np

def calculate_cost(schedule: Dict[int, int], P_arr: np.ndarray, L_arr: np.ndarray, Ci_arr: np.ndarray, Ck_arr: np.ndarray) -> float:
    """
    Calcula el costo total de un horario dado, incluyendo penalizaciones por violación de L[k].
    """
    if not schedule: # Si no hay horario, el costo es infinito
        return float('inf')

    plane_indices = np.array(list(schedule.keys()))
    landing_times = np.array(list(schedule.values()))

    # Validación de índices para evitar errores en producción
    if np.any(plane_indices < 0) or \
       np.any(plane_indices >= len(P_arr)) or \
       np.any(plane_indices >= len(L_arr)) or \
       np.any(plane_indices >= len(Ci_arr)) or \
       np.any(plane_indices >= len(Ck_arr)):
        print(f"Error crítico en calculate_cost: Índices de avión fuera de rango en el horario.")
        return float('inf') # Costo infinito para un horario inválido

    P_scheduled = P_arr[plane_indices]
    Ci_scheduled = Ci_arr[plane_indices]
    Ck_scheduled = Ck_arr[plane_indices]

    early_penalty = np.sum(Ci_scheduled * np.maximum(0, P_scheduled - landing_times))
    late_penalty = np.sum(Ck_scheduled * np.maximum(0, landing_times - P_scheduled))
    current_total_cost = early_penalty + late_penalty

    # Añadir penalización por violación de L[k]
    L_scheduled = L_arr[plane_indices]
    l_violation_units = np.maximum(0, landing_times - L_scheduled)
    l_violation_cost = np.sum(L_VIOLATION_PENALTY_PER_UNIT * l_violation_units)
    current_total_cost += l_violation_cost
    
    return current_total_cost

Schedule = Dict[int, int]
RunwayAssignment = Dict[int, int]

def greedy_deterministic(num_planes: int, E_arr: np.ndarray, P_arr: np.ndarray, L_arr_param: np.ndarray,
                         Ci_arr: np.ndarray, Ck_arr: np.ndarray, tau_arr: np.ndarray,
                         num_runways: int = 1,
                         force_schedule_despite_L: bool = False) -> Tuple[Optional[Schedule], Optional[RunwayAssignment], float, bool]:
    plane_indices_sorted = sorted(range(num_planes), key=lambda k_idx: (P_arr[k_idx], E_arr[k_idx]))

    schedule: Schedule = {}
    runway_assignment: RunwayAssignment = {}
    
    last_plane_idx_on_runway: List[int] = [-1] * num_runways
    last_landing_time_on_runway: List[int] = [-1] * num_runways
    
    all_planes_scheduled = True
    for k in plane_indices_sorted:
        best_landing_time_for_k = float('inf')
        best_runway_for_k = -1
        
        for runway_idx in range(num_runways):
            min_start_time = E_arr[k]
            earliest_after_separation = 0
            
            if last_plane_idx_on_runway[runway_idx] != -1:
                separation_needed = tau_arr[last_plane_idx_on_runway[runway_idx]][k]
                if separation_needed >= 99999:
                    continue 
                earliest_after_separation = last_landing_time_on_runway[runway_idx] + separation_needed
            
            actual_landing_time = max(min_start_time, earliest_after_separation)
            
            can_schedule_this_way = False
            if actual_landing_time <= L_arr_param[k]:
                can_schedule_this_way = True
            elif force_schedule_despite_L: # Si se fuerza, se permite violar L[k]
                can_schedule_this_way = True

            if can_schedule_this_way and actual_landing_time < best_landing_time_for_k:
                best_landing_time_for_k = actual_landing_time
                best_runway_for_k = runway_idx
        
        if best_runway_for_k == -1:
            # No se pudo programar este avión, ni siquiera forzando L[k] (quizás por tau o E[k])
            all_planes_scheduled = False
            break # Salir del bucle de aviones, no se puede continuar
        
        schedule[k] = best_landing_time_for_k
        runway_assignment[k] = best_runway_for_k
        last_plane_idx_on_runway[best_runway_for_k] = k
        last_landing_time_on_runway[best_runway_for_k] = best_landing_time_for_k
    
    if not all_planes_scheduled or len(schedule) != num_planes :
        # Si no todos los aviones fueron programados, la solución no es completa.
        return None, None, float('inf'), False # False para estrictamente factible

    total_cost = calculate_cost(schedule, P_arr, L_arr_param, Ci_arr, Ck_arr)

    is_strictly_feasible = True
    for plane_idx_check, landing_time_check in schedule.items():
        if landing_time_check > L_arr_param[plane_idx_check]:
            is_strictly_feasible = False
            break
                
    return schedule, runway_assignment, total_cost, is_strictly_feasible

def greedy_stochastic(num_planes: int, E_arr: np.ndarray, P_arr: np.ndarray, L_arr_param: np.ndarray,
                       Ci_arr: np.ndarray, Ck_arr: np.ndarray, tau_arr: np.ndarray,
                       num_runways: int = 1, rcl_size: int = 3,
                       seed: Optional[int] = None,
                       force_schedule_despite_L: bool = False) -> Tuple[Optional[Schedule], Optional[RunwayAssignment], float, bool]:
    if seed is not None:
        random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    schedule: Schedule = {}
    runway_assignment: RunwayAssignment = {}
    
    last_plane_idx_on_runway: List[int] = [-1] * num_runways
    last_landing_time_on_runway: List[int] = [-1] * num_runways

    all_planes_scheduled_flag = True
    while unscheduled_planes:
        candidates: List[Tuple[int, int, int]] = []

        for k in unscheduled_planes:
            for runway_idx in range(num_runways):
                min_start_time = E_arr[k]
                earliest_after_separation = 0
                
                if last_plane_idx_on_runway[runway_idx] != -1:
                    separation_needed = tau_arr[last_plane_idx_on_runway[runway_idx]][k]
                    if separation_needed >= 99999:
                        continue
                    earliest_after_separation = last_landing_time_on_runway[runway_idx] + separation_needed
                
                earliest_possible_time = max(min_start_time, earliest_after_separation)
                
                if earliest_possible_time <= L_arr_param[k]:
                    candidates.append((k, runway_idx, earliest_possible_time))
                elif force_schedule_despite_L: # Si se fuerza, se permite violar L[k]
                    candidates.append((k, runway_idx, earliest_possible_time))

        if not candidates:
            # No se pudo programar ninguno de los aviones restantes
            all_planes_scheduled_flag = False
            break # Salir del bucle while
        
        candidates.sort(key=lambda x: x[2])
        current_rcl_size = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_size]
        
        chosen_plane_idx, chosen_runway, chosen_landing_time = random.choice(rcl)

        schedule[chosen_plane_idx] = chosen_landing_time
        runway_assignment[chosen_plane_idx] = chosen_runway
        last_plane_idx_on_runway[chosen_runway] = chosen_plane_idx
        last_landing_time_on_runway[chosen_runway] = chosen_landing_time
        unscheduled_planes.remove(chosen_plane_idx)

    if not all_planes_scheduled_flag or len(schedule) != num_planes:
        return None, None, float('inf'), False

    total_cost = calculate_cost(schedule, P_arr, L_arr_param, Ci_arr, Ck_arr)

    is_strictly_feasible = True
    for plane_idx_check, landing_time_check in schedule.items():
        if landing_time_check > L_arr_param[plane_idx_check]:
            is_strictly_feasible = False
            break
                
    return schedule, runway_assignment, total_cost, is_strictly_feasible

def print_solution(schedule: Optional[Schedule], runway_assignment: Optional[RunwayAssignment],
                   total_cost: float, strictly_feasible: bool, num_runways: int, L_arr_param: np.ndarray,
                   was_forced: bool = False):
    if schedule is None or not schedule:
        print("No se generó un horario (o está vacío).")
        if was_forced:
            print("Incluso forzando violaciones de L[k], no se pudo completar un horario para todos los aviones.")
        return

    if not strictly_feasible:
        print("ADVERTENCIA: La solución NO es estrictamente factible (puede violar L[k] o no incluir todos los aviones).")
        if was_forced:
            print("Se permitió la violación de L[k] para generar este horario.")
    
    print(f"Costo total (puede incluir penalizaciones por violar L[k]): {total_cost:.2f}")
    print("Avión\tPista\tTiempo Aterrizaje\tL[k]\tViola L[k]?")
    print("-" * 60) # Ajustado para nueva columna

    sorted_schedule_items = []
    # Asegurarse de que solo procesamos aviones que están en el horario
    # (aunque si schedule no es None, deberían estar todos si la lógica es correcta)
    for plane_idx in schedule.keys():
        landing_time = schedule[plane_idx]
        runway = runway_assignment[plane_idx]
        sorted_schedule_items.append((plane_idx, runway, landing_time))
    
    sorted_schedule_items.sort(key=lambda x: (x[1], x[2])) # Ordenar por pista, luego por tiempo

    for plane_idx, runway, landing_time in sorted_schedule_items:
        l_k_value = L_arr_param[plane_idx]
        l_violated_char = "Sí" if landing_time > l_k_value else "No"
        # Usar índice base 1 para aviones y pistas para mayor claridad
        print(f"{plane_idx + 1}\t{runway + 1}\t{landing_time}\t\t\t{l_k_value}\t{l_violated_char}")
    
    if num_runways > 1:
        print("\nEstadísticas por pista:")
        for runway_idx_print in range(num_runways):
            planes_in_this_runway = [p_idx for p_idx, r_idx, _ in sorted_schedule_items if r_idx == runway_idx_print]
            print(f"Pista {runway_idx_print + 1}: {len(planes_in_this_runway)} aviones")

def run_multiple_stochastic(num_planes: int, E_arr: np.ndarray, P_arr: np.ndarray, L_arr_param: np.ndarray,
                            Ci_arr: np.ndarray, Ck_arr: np.ndarray, tau_arr: np.ndarray,
                            num_runways: int = 1, num_runs: int = 10, rcl_size: int = 3,
                            force_schedule_despite_L: bool = False) -> List[Dict[str, Any]]:
    results = []
    for i in range(num_runs):
        seed = i 
        start_time = time.time()
        schedule, runway_assignment, cost, strictly_feasible_flag = greedy_stochastic(
            num_planes, E_arr, P_arr, L_arr_param, Ci_arr, Ck_arr, tau_arr, 
            num_runways=num_runways, rcl_size=rcl_size, seed=seed,
            force_schedule_despite_L=force_schedule_despite_L
        )
        end_time = time.time()

        results.append({
            "seed": seed,
            "cost": cost,
            "time": end_time - start_time,
            "schedule": schedule, # Puede ser None si no se programaron todos
            "runway_assignment": runway_assignment, # Puede ser None
            "feasible": strictly_feasible_flag # True si T[k] <= L[k] para todos Y todos programados
        })
    return results

def main():
    DEFAULT_CASE_DIR = "casos"

    select_str = input("Selecciona el caso a evaluar (1-4) o 0 para salir: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n")
    try:
        select = int(select_str)
        if select == 0:
            print("Saliendo...")
            sys.exit()
        if not 1 <= select <= 4:
            print("Opción no válida. Por favor, elija entre 1 y 4.")
            sys.exit()
    except ValueError:
        print("Entrada inválida. Por favor ingrese un número.")
        sys.exit()

    num_runways_str = input("\nSelecciona el número de pistas de aterrizaje (1-2): ")
    try:
        num_runways = int(num_runways_str)
        if not 1 <= num_runways <= 2:
            print("Número de pistas no válido. Se usará 1 pista por defecto.")
            num_runways = 1
    except ValueError:
        print("Entrada inválida. Se usará 1 pista por defecto.")
        num_runways = 1

    # Forzar violación de L[k] (con penalización) si es el Caso 3
    force_L_violations_if_needed = (select == 3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, DEFAULT_CASE_DIR, f'case{select}.txt')

    print(f"\nBuscando el archivo en: {filename}")
    print(f"Utilizando {num_runways} pista(s) de aterrizaje.")
    if force_L_violations_if_needed:
        print(f"NOTA: Para el CASO {select}, se intentará forzar la programación incluso si se viola L[k] (con alta penalización).")

    try:
        num_planes, E_data, P_data, L_data, Ci_data, Ck_data, tau_data = read_file(filename)
        print("-" * 40)
        print("Lectura del archivo exitosa.")
        print(f"Número de aviones (D): {num_planes}")
        # (Opcional: imprimir E_data, P_data, L_data, Ci_data, Ck_data, tau_data)
        print("-" * 40)

        # --- Greedy Determinista ---
        print("\n" + "=" * 40)
        print(f"ALGORITMO GREEDY DETERMINISTA ({num_runways} PISTA(S))")
        print("=" * 40)

        start_time_det = time.time()
        det_schedule, det_runway_assignment, det_cost, det_strictly_feasible = greedy_deterministic(
            num_planes, E_data, P_data, L_data, Ci_data, Ck_data, tau_data, num_runways=num_runways,
            force_schedule_despite_L=force_L_violations_if_needed
        )
        end_time_det = time.time()

        print(f"Tiempo de ejecución: {end_time_det - start_time_det:.4f} segundos")
        print_solution(det_schedule, det_runway_assignment, det_cost, det_strictly_feasible, num_runways, L_data, force_L_violations_if_needed)

        # --- Greedy Estocástico ---
        print("\n" + "=" * 40)
        print(f"ALGORITMO GREEDY ESTOCÁSTICO ({num_runways} PISTA(S), 10 EJECUCIONES)")
        print("=" * 40)

        stochastic_results = run_multiple_stochastic(
            num_planes, E_data, P_data, L_data, Ci_data, Ck_data, tau_data,
            num_runways=num_runways, num_runs=10, rcl_size=3,
            force_schedule_despite_L=force_L_violations_if_needed
        )

        print("\nResumen de resultados estocásticos:")
        print("Seed\tCosto Total\t\tTiempo (s)\tEstrict. Factible?")
        print("-" * 60) # Ajustado

        for result in stochastic_results:
            cost_str = f"{result['cost']:.2f}" if result['schedule'] is not None else "inf (No Schedule)"
            feasible_char = "Sí" if result['feasible'] else "No"
            print(f"{result['seed']}\t{cost_str}\t\t{result['time']:.4f}\t\t{feasible_char}")

        # --- Encontrar y Mostrar la Mejor Solución Estocástica ---
        schedules_found_stochastic = [res for res in stochastic_results if res['schedule'] is not None]
        best_stochastic_result = None

        if schedules_found_stochastic:
            strictly_feasible_ones = [res for res in schedules_found_stochastic if res['feasible']]
            if strictly_feasible_ones:
                best_stochastic_result = min(strictly_feasible_ones, key=lambda x: x['cost'])
            elif force_L_violations_if_needed: # Si no hay estrictamente factibles Y se forzó L
                best_stochastic_result = min(schedules_found_stochastic, key=lambda x: x['cost'])
        
        if best_stochastic_result:
            print("\nMejor solución estocástica encontrada (considerando forzar L[k] si fue necesario):")
            print(f"Seed: {best_stochastic_result['seed']}")
            print(f"Costo: {best_stochastic_result['cost']:.2f}")
            print(f"Tiempo: {best_stochastic_result['time']:.4f} segundos")
            print(f"Estrictamente Factible: {'Sí' if best_stochastic_result['feasible'] else 'No'}")
            print("\nDetalle de la mejor solución estocástica:")
            print_solution(
                best_stochastic_result['schedule'],
                best_stochastic_result['runway_assignment'],
                best_stochastic_result['cost'],
                best_stochastic_result['feasible'],
                num_runways, L_data,
                force_L_violations_if_needed
            )
        else:
            print("\nMejor solución estocástica:")
            print("No se encontraron programaciones completas en las ejecuciones estocásticas.")
            
        # --- Comparación Final ---
        print("\n" + "=" * 40)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 40)

        det_cost_display = f"{det_cost:.2f}" if det_schedule is not None else "inf (Sin Horario)"
        det_feasible_display = 'Sí' if det_strictly_feasible else 'No'
        
        sto_cost_display = "inf (Sin Horario)"
        sto_feasible_display = "N/A"
        if best_stochastic_result and best_stochastic_result['schedule'] is not None:
            sto_cost_display = f"{best_stochastic_result['cost']:.2f}"
            sto_feasible_display = 'Sí' if best_stochastic_result['feasible'] else 'No'

        print(f"Costo greedy determinista: {det_cost_display} (Estrict. Factible: {det_feasible_display})")
        print(f"Costo mejor greedy estocástico: {sto_cost_display} (Estrict. Factible: {sto_feasible_display})")

        if det_schedule is not None and best_stochastic_result and best_stochastic_result['schedule'] is not None:
            diff = det_cost - best_stochastic_result['cost']
            perc_improvement = 0
            # Evitar división por cero o por infinito si el costo determinista es muy bajo o inf
            if abs(det_cost) > 1e-9 and det_cost != float('inf'): 
                perc_improvement = (diff / abs(det_cost)) * 100 
            elif diff > 0 : perc_improvement = float('inf')
            elif diff < 0 : perc_improvement = float('-inf')


            print(f"Diferencia (Det - Sto): {diff:.2f}")
            print(f"Mejora porcentual del estocástico sobre determinista: {perc_improvement:.2f}%")
        elif det_schedule is None and best_stochastic_result and best_stochastic_result['schedule'] is not None:
            print("Determinista no encontró horario, Estocástico sí.")
        elif det_schedule is not None and (best_stochastic_result is None or best_stochastic_result['schedule'] is None):
            print("Estocástico no encontró horario, Determinista sí.")
        else:
            print("Ninguno de los algoritmos encontró un horario completo.")
            
        print("=" * 40)

    except FileNotFoundError: # Ya manejado en read_file, pero por si acaso.
        print(f"Error Crítico: No se encontró el archivo de datos {filename}")
    except Exception as e:
        print(f"Error Crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()