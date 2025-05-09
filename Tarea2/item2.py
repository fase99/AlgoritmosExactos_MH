import numpy as np
import os
import sys
import random
from typing import List, Tuple, Dict, Optional, Any
import time
import copy
import matplotlib.pyplot as plt 


PlaneIdx = int
PistIdx = int


InitialSchedule = Dict[PlaneIdx, int]
Initial_pist = Dict[PlaneIdx, PistIdx]

#conf para HC
PistSequence = List[PlaneIdx]
PistLayout = Dict[PistIdx, PistSequence] 
FinalSchedule = Dict[PlaneIdx, int]

def read(path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lee los datos de la instancia desde el archivo."""

    with open(path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    line_idx = 0

    try:
        if line_idx >= len(lines): 
            raise ValueError("Archivo vacío.")
        

        num_planes = int(lines[line_idx])
        line_idx += 1
        E, P, L, Ci, Ck, tau = [], [], [], [], [], []

        for i in range(num_planes):
            if line_idx >= len(lines): 
                raise ValueError(f"EOF datos avión {i}.")
            

            parts = lines[line_idx].split()
            line_idx += 1

            if len(parts) != 5: 
                raise ValueError(f"Datos avión {i}: 5 valores esperados, {len(parts)} obtenidos.")
            
            E.append(int(parts[0])); P.append(int(parts[1])); L.append(int(parts[2]))
            Ci.append(float(parts[3])); Ck.append(float(parts[4]))

            tau_current = []
            while len(tau_current) < num_planes:

                if line_idx >= len(lines): 
                    raise ValueError(f"EOF sep. avión {i}.")
                
                sep_parts = lines[line_idx].split(); line_idx += 1
                tau_current.extend([int(p) for p in sep_parts])

            if len(tau_current) != num_planes: 
                raise ValueError(f"Sep. avión {i}: {num_planes} esperados, {len(tau_current)} obtenidos.")
            
            tau.append(tau_current)

        if len(E) != num_planes: 
            raise ValueError("Cuenta de datos de avión incorrecta.")
        
        E = np.array(E); P_val = np.array(P); L = np.array(L)
        Ci = np.array(Ci); Ck = np.array(Ck); tau = np.array(tau)


        if not (np.all(E <= P_val) and np.all(P_val <= L)): 
            print("Advertencia: E <= P <= L no se cumple.")

        return num_planes, E, P_val, L, Ci, Ck, tau
    
    except Exception as e: print(f"Error leyendo archivo '{path}': {e}"); sys.exit(1)

def calc_cost(schedule: InitialSchedule, P_val: np.ndarray, Ci: np.ndarray, Ck: np.ndarray) -> float:
    #Calcula el costo total de penalización para un horario dado.

    if not schedule: 
        return float('inf')
    
    plane_indices = np.array(list(schedule.keys()))
    landing_times = np.array(list(schedule.values()))

    if np.any(plane_indices < 0) or np.any(plane_indices >= len(P_val)): 
        return float('inf')
    
    P = P_val[plane_indices]
    Ci = Ci[plane_indices]
    Ck = Ck[plane_indices]
    antes = np.sum(Ci * np.maximum(0, P - landing_times))
    tarde = np.sum(Ck * np.maximum(0, landing_times - P))
    cost = antes + tarde

    return cost if not (np.isnan(cost) or np.isinf(cost)) else float('inf')

def greedy_stochastic(
    num_planes: int, E: np.ndarray, P_val: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    num_pist: int = 1, rcl_size: int = 3,
    seed: Optional[int] = None
) -> Tuple[Optional[InitialSchedule], Optional[Initial_pist], float, bool]:
    

    if seed is not None: random.seed(seed)
    planes_noScheduled = set(range(num_planes))
    schedule: InitialSchedule = {}
    pist_assign: Initial_pist = {}
    last_plane_idx: List[int] = [-1] * num_pist
    last_landing_time: List[int] = [-1] * num_pist
    count = 0


    while planes_noScheduled:
        candidates: List[Tuple[int, int, int]] = []

        for k in planes_noScheduled:
            for pist_idx in range(num_pist):
                min_start_time = E[k]
                e_after_sep = 0

                if last_plane_idx[pist_idx] != -1:

                    sep_needed = tau[last_plane_idx[pist_idx]][k]

                    if sep_needed >= 99999: 
                        continue


                    e_after_sep = last_landing_time[pist_idx] + sep_needed

                earliestTime = max(min_start_time, e_after_sep)

                if earliestTime <= L[k]: 
                    candidates.append((k, pist_idx, earliestTime))

        if not candidates: 
            return None, None, float('inf'), False
        
        candidates.sort(key=lambda x: x[2])
        current_rcl_size = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_size]

        if not rcl: 
            return None, None, float('inf'), False
        chosen_plane_idx, chosen_runway, chosen_landing_time = random.choice(rcl)

        schedule[chosen_plane_idx] = chosen_landing_time
        pist_assign[chosen_plane_idx] = chosen_runway
        last_plane_idx[chosen_runway] = chosen_plane_idx
        last_landing_time[chosen_runway] = chosen_landing_time
        planes_noScheduled.remove(chosen_plane_idx)
        count += 1


    if count != num_planes: 
        return None, None, float('inf'), False
    total_cost = calc_cost(schedule, P_val, Ci, Ck) 
    return schedule, pist_assign, total_cost, True

def greedy_deterministic( 
    num_planes: int, E: np.ndarray, P_val: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    num_runways: int = 1
) -> Tuple[Optional[InitialSchedule], Optional[Initial_pist], float, bool]:
    

    plane_indices = sorted(range(num_planes), key=lambda k: (P_val[k], E[k]))
    schedule: InitialSchedule = {}
    pist_assign: Initial_pist = {}

    last_plane_idx: List[int] = [-1] * num_runways
    last_landing_time: List[int] = [-1] * num_runways

    for k in plane_indices:
        best_landing_time = float('inf')
        best_pist = -1

        for runway_idx in range(num_runways):
            min_start_time = E[k]
            e_after_sep = 0

            if last_plane_idx[runway_idx] != -1:
                separation_needed = tau[last_plane_idx[runway_idx]][k]

                if separation_needed >= 99999: 
                    continue

                e_after_sep = last_landing_time[runway_idx] + separation_needed


            actual_landing_time = max(min_start_time, e_after_sep)

            if actual_landing_time <= L[k] and actual_landing_time < best_landing_time:
                best_landing_time = actual_landing_time; best_pist = runway_idx

        if best_pist == -1: 
            return None, None, float('inf'), False
        
        schedule[k] = best_landing_time
        pist_assign[k] = best_pist
        last_plane_idx[best_pist] = k
        last_landing_time[best_pist] = best_landing_time

    total_cost = calc_cost(schedule, P_val, Ci, Ck)
    return schedule, pist_assign, total_cost, True

def pist_layout_convert(
    initial_schedule: Optional[InitialSchedule], initial_assignment: Optional[Initial_pist],
    num_planes: int, num_pist: int
) -> Optional[PistLayout]:
    

    if initial_schedule is None or initial_assignment is None: 
        return None
    

    layout: PistLayout = {p: [] for p in range(num_pist)}
    details = []


    for p_idx, time_val in initial_schedule.items():
        if p_idx not in initial_assignment: return None
        details.append((p_idx, initial_assignment[p_idx], time_val))
    details.sort(key=lambda x: (x[1], x[2]))
    planes_count = 0

    for p_idx, pist_idx_val, _ in details:
        if pist_idx_val >= num_pist: 
            return None
        
        layout[pist_idx_val].append(p_idx)
        planes_count +=1

    if planes_count != num_planes: 
        return None
    
    return layout

def final_time(
    runway_layout: PistLayout, num_pist: int, E: np.ndarray, L: np.ndarray, tau: np.ndarray
) -> Tuple[Optional[FinalSchedule], bool]:
    

    schedule: FinalSchedule = {}; last_time: Dict[PistIdx, int] = {r: -1 for r in range(num_pist)}
    last_plane: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_pist)}; processed = set()


    for p_idx in range(num_pist):

        if p_idx not in runway_layout: runway_layout[p_idx] = []

        for plane in runway_layout[p_idx]:
            if plane in processed: 
                return None, False
            
            processed.add(plane)

            if not (0 <= plane < len(E)): 
                return None, False
            
            start = E[plane]
            sep_delay = 0

            if last_plane[p_idx] != -1:

                sep = tau[last_plane[p_idx]][plane]

                if sep >= 99999: 
                    return None, False
                
                sep_delay = last_time[p_idx] + sep
                
            land_time = max(start, sep_delay)

            if land_time > L[plane]:
                return None, False
            schedule[plane] = land_time
            last_time[p_idx] = land_time
            last_plane[p_idx] = plane

    if len(processed) != len(E): 
        return None, False
    
    return schedule, True

def ev_pist_layout(
    runway_layout: PistLayout, num_pist: int, E: np.ndarray, P_val: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray
) -> Tuple[float, Optional[FinalSchedule], bool]:
    

    final_schedule, feasible = final_time(runway_layout, num_pist, E, L, tau)

    if not feasible or final_schedule is None: 
        return float('inf'), None, False
    
    # Asegurar que el schedule final es completo para la función de costo

    if len(final_schedule) != len(E): 
        return float('inf'), None, False
    
    cost = calc_cost(final_schedule, P_val, Ci, Ck)

    return cost, final_schedule, feasible

def hill_climbing(
    initial_layout: PistLayout, num_pist: int, E: np.ndarray, P_val: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, use_best_improvement: bool
) -> Tuple[PistLayout, float]:
    

    current_layout = copy.deepcopy(initial_layout)
    current_cost, _, feasible = ev_pist_layout(current_layout, num_pist, E, P_val, L, Ci, Ck, tau)


    if not feasible: 
        return initial_layout, float('inf')
    

    while True:
        best_neighbor_layout = None
        best_neighbor_cost = current_cost
        moved = False

        for r_idx in range(num_pist): # Intra-Runway Swaps
            seq = current_layout[r_idx]


            if len(seq) < 2: 
                continue
            for i in range(len(seq)):

                for j in range(i + 1, len(seq)):

                    n_layout = copy.deepcopy(current_layout)
                    n_layout[r_idx][i], n_layout[r_idx][j] = n_layout[r_idx][j], n_layout[r_idx][i]
                    cost_n, _, feas_n = ev_pist_layout(n_layout, num_pist, E, P_val, L, Ci, Ck, tau)

                    if feas_n and cost_n < best_neighbor_cost:

                        if use_best_improvement: 
                            best_neighbor_cost,best_neighbor_layout,moved = cost_n,n_layout,True

                        else: 
                            current_layout,current_cost,moved = n_layout,cost_n,True
                            break

                if not use_best_improvement and moved: 
                    break
            if not use_best_improvement and moved: 
                break
        if not use_best_improvement and moved: 
            continue


        if num_pist > 1: # Inter-Runway Moves
            for src_r in range(num_pist):
                if not current_layout[src_r]: 
                    continue


                for plane_pos in range(len(current_layout[src_r])):

                    if plane_pos >= len(current_layout[src_r]): 
                        continue


                    for dest_r in range(num_pist):

                        if src_r == dest_r: 
                            continue
                        n_layout = copy.deepcopy(current_layout)


                        try: 
                            m_plane = n_layout[src_r].pop(plane_pos)

                        except IndexError: 
                            continue

                        n_layout[dest_r].append(m_plane)
                        cost_n, _, feas_n = ev_pist_layout(n_layout, num_pist, E, P_val, L, Ci, Ck, tau)

                        if feas_n and cost_n < best_neighbor_cost:

                            if use_best_improvement: 
                                best_neighbor_cost,best_neighbor_layout,moved = cost_n,n_layout,True
                            
                            else: 
                                current_layout,current_cost,moved = n_layout,cost_n,True
                                break

                    if not use_best_improvement and moved: 
                        break

                if not use_best_improvement and moved: 
                    break
            if not use_best_improvement and moved: 
                continue

        if use_best_improvement:
            if moved and best_neighbor_layout: 
                current_layout,current_cost = best_neighbor_layout,best_neighbor_cost

            else: 
                break
        else:
            if not moved: 
                break
    return current_layout, current_cost

# --- Fase de Construcción GRASP (Clásica, incremental con RCL) ---
def grasp_construct(
    num_planes: int, num_pist: int, E: np.ndarray, P_val: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, alpha_rcl: float
) -> Optional[PistLayout]:
    

    unscheduled = set(range(num_planes))
    layout: PistLayout = {r: [] for r in range(num_pist)}
    last_time: Dict[PistIdx, int] = {r: -1 for r in range(num_pist)}
    last_plane: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_pist)}
    scheduled_count = 0


    while unscheduled:
        candidates = []
        min_ct = float('inf')
        max_ct_rcl =  -float('inf')

        for k_p in unscheduled:

            for r_i in range(num_pist):
                start = E[k_p]
                sep_d = 0
                prev_p = last_plane[r_i]

                if prev_p != -1:
                    sep = tau[prev_p][k_p]
                    if sep >= 99999: continue
                    sep_d = last_time[r_i] + sep

                p_time = max(start, sep_d)

                if p_time <= L[k_p]:
                    candidates.append({'plane':k_p, 'runway':r_i, 'time':p_time})
                    min_ct = min(min_ct, p_time); max_ct_rcl = max(max_ct_rcl, p_time)

        if not candidates: return None
        thresh = min_ct + alpha_rcl * (max_ct_rcl - min_ct) if max_ct_rcl > min_ct else min_ct
        rcl = [c for c in candidates if c['time'] <= thresh]

        if not rcl: rcl = [c for c in candidates if c['time'] == min_ct]
        if not rcl: return None

        chosen = random.choice(rcl)
        c_p, c_r, c_t = chosen['plane'], chosen['runway'], chosen['time']
        layout[c_r].append(c_p)
        last_time[c_r] = c_t
        last_plane[c_r] = c_p
        unscheduled.remove(c_p)
        scheduled_count +=1

    if scheduled_count != num_planes: 
        return None
    
    return layout

# --- Algoritmo GRASP Clásico con Warm Start ---
def grasp_algorithm(
    num_planes: int, num_pist: int, E: np.ndarray, P: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    alpha_rcl_grasp: float, max_grasp_iterations: int, use_best_improvement_hc: bool,
    num_stochastic_warm_starts: int, rcl_size_stochastic_warm_start: int,
    max_restarts_in_warm_start_hc: int # Para el "restart" dentro del warm start
) -> Tuple[Optional[PistLayout], float, List[float]]:
    best_overall_layout: Optional[PistLayout] = None
    best_overall_cost = float('inf'); cost_history = []
    start_time = time.time()


    hc_name = "Mejor Mejora" if use_best_improvement_hc else "Alguna Mejora"

    print(f"\n--- GRASP (HC: {hc_name}, Iter GRASP: {max_grasp_iterations}, Alpha GRASP: {alpha_rcl_grasp}) ---")
    print(f"Warm Start: {num_stochastic_warm_starts} ejec. Greedy Stoch. (RCL={rcl_size_stochastic_warm_start})")
    print(f"  con hasta {max_restarts_in_warm_start_hc} restarts internos si HC de warm start se estanca.")


    print(f"\n[GRASP Warm Start] Mejorando {num_stochastic_warm_starts} soluciones del Greedy Estocástico...")


    for i_ws in range(num_stochastic_warm_starts):

        best_layout_this_ws_run: Optional[PistLayout] = None
        best_cost_this_ws_run = float('inf')

        for hc_restart_attempt in range(max_restarts_in_warm_start_hc + 1):

            seed_gs = i_ws * (max_restarts_in_warm_start_hc + 101) + hc_restart_attempt
            
            initial_sched, initial_assign, _, feasible_gs = greedy_stochastic(
                num_planes, E, P, L, Ci, Ck, tau, num_pist, rcl_size_stochastic_warm_start, seed_gs)
            if not feasible_gs or initial_sched is None or initial_assign is None: continue
            
            base_layout = pist_layout_convert(initial_sched, initial_assign, num_planes, num_pist)
            if base_layout is None: continue
            
            cost_base_eval,_,feas_base_eval=ev_pist_layout(base_layout,num_pist,E,P,L,Ci,Ck,tau)
            if not feas_base_eval: continue

            hc_layout, hc_cost = hill_climbing(base_layout, num_pist,E,P,L,Ci,Ck,tau,use_best_improvement_hc)
            if hc_cost < best_cost_this_ws_run:
                best_cost_this_ws_run = hc_cost
                best_layout_this_ws_run = hc_layout
        
        if best_layout_this_ws_run and best_cost_this_ws_run < best_overall_cost:
            best_overall_cost = best_cost_this_ws_run 
            best_overall_layout = best_layout_this_ws_run
            print(f"    Mejor desde Warm Start Ejec. {i_ws+1} (tras restarts HC int.): {best_overall_cost:.2f}")
    
    if best_overall_cost != float('inf'): cost_history.append(best_overall_cost)
    print(f"[GRASP Warm Start] Mejor costo tras fase Warm Start: {best_overall_cost if best_overall_cost != float('inf') else 'inf':.2f}")

    print(f"\n[GRASP] Iniciando {max_grasp_iterations} iteraciones principales...")
    for iter_g in range(max_grasp_iterations):

        constructed_layout = grasp_construct(num_planes,num_pist,E,P,L,Ci,Ck,tau,alpha_rcl_grasp)
        if constructed_layout is None or sum(len(s) for s in constructed_layout.values()) != num_planes:
            if best_overall_cost!=float('inf'): cost_history.append(best_overall_cost)
            elif cost_history: cost_history.append(cost_history[-1])
            continue

        hc_layout, hc_cost = hill_climbing(constructed_layout,num_pist,E,P,L,Ci,Ck,tau,use_best_improvement_hc)
        if hc_cost < best_overall_cost:
            best_overall_cost = hc_cost; best_overall_layout = hc_layout
            print(f"    >> ¡Nuevo MEJOR GLOBAL en iter. GRASP {iter_g+1}!: {best_overall_cost:.2f}")

        if best_overall_cost!=float('inf'): 
            cost_history.append(best_overall_cost)

        elif cost_history: 
            cost_history.append(cost_history[-1])
    
    total_time = time.time() - start_time

    print(f"\n[GRASP Final] Proceso ({hc_name}) finalizado en {total_time:.4f}s.")


    if best_overall_layout is None: 
        return None, float('inf'), cost_history
    final_c,_,final_f = ev_pist_layout(best_overall_layout,num_pist,E,P,L,Ci,Ck,tau)

    if not final_f: 
        return None,float('inf'),cost_history
    
    if abs(final_c - best_overall_cost) > 1e-5: 
        best_overall_cost = final_c

    print(f"[GRASP Final] Mejor costo verificado: {best_overall_cost:.2f}")
    return best_overall_layout, best_overall_cost, cost_history

def print_runway_solution(
    final_schedule: Optional[FinalSchedule], total_cost: float, is_feasible: bool,
    runway_layout: Optional[PistLayout] = None, num_pist: Optional[int] = None):

    
    cost_str = "inf" if total_cost == float('inf') else f"{total_cost:.2f}"
    if not is_feasible or final_schedule is None: 
        print(f"No se encontró solución factible. Costo: {cost_str}") 
        return
    

    print(f"Costo total: {total_cost:.2f}")
    if runway_layout and num_pist is not None:
        print("\nAsignación y Tiempos por Pista:") 

        details_by_runway = {r:[] for r in range(num_pist)}
        plane_to_runway: Dict[PlaneIdx, PistIdx] = {}


        for r_idx, planes in runway_layout.items():
            if r_idx >= num_pist: 
                continue

            for p_idx in planes: 
                plane_to_runway[p_idx] = r_idx


        sorted_by_time = sorted(final_schedule.items(), key=lambda item: item[1])


        for p_idx,l_time in sorted_by_time:
            if p_idx in plane_to_runway: 
                details_by_runway[plane_to_runway[p_idx]].append(f"Avión {p_idx+1}@{l_time}")

        for r_idx in range(num_pist): 
            print(f"  Pista {r_idx+1}: [{', '.join(details_by_runway[r_idx])}]")

    print("\nHorario General Ordenado:")
    print("Avión\tTiempo de aterrizaje")
    print("-" * 25)

    sorted_final_sched = sorted(final_schedule.items(), key=lambda item: item[1])
    for p_idx, l_time in sorted_final_sched: 
        print(f"{p_idx + 1}\t{l_time}")

def main():
    DEFAULT_CASE_DIR = "casos"

    while True:
        select_str = input("Selecciona caso (1-4) o 0 salir: \n1. Case1 \n2. Case2 \n3. Case3 \n4. Case4 \n0. Salir\n")
        try: 
            select = int(select_str);
            if select == 0: 
                print("Saliendo...")
                sys.exit()
            if 1 <= select <= 4: 
                break

            else: print("Opción no válida.")

        except ValueError: print("Entrada inválida.")


    while True:
        try: 
            num_runways_str = input("Pistas (1 o 2): ")
            num_pists = int(num_runways_str)

            if num_pists in [1,2]: 
                break
            else: 
                print("Pistas debe ser 1 o 2.")

        except ValueError: 
            print("Número entero (1 o 2).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir if script_dir else os.getcwd()
    filename = os.path.join(base_dir,DEFAULT_CASE_DIR,f'case{select}.txt')

    print(f"\nLeyendo: {filename}")

    try:
        num_planes, E, P_array, L, Ci, Ck, tau = read(filename) # Usa la nueva función de carga
        print("-" * 40 + f"\nLectura OK. Aviones: {num_planes}, Pistas: {num_pists}" + "\n" + "-" * 40)

        # Parámetros para GRASP
        num_stoch_warm_starts_str = input(f"Num. ejecuciones Greedy Estocástico para Warm Start (ej: 5): ")
        NUM_STOCH_WARM_STARTS = int(num_stoch_warm_starts_str) if num_stoch_warm_starts_str.isdigit() and int(num_stoch_warm_starts_str) >=0 else 5
        
        restarts_hc_ws_str = input(f"Max restarts HC interno en Warm Start (ej: 0, 1): ")
        MAX_RESTARTS_HC_WARM_START = int(restarts_hc_ws_str) if restarts_hc_ws_str.isdigit() and int(restarts_hc_ws_str) >=0 else 0
        
        iter_grasp_str = input(f"Num. iteraciones GRASP Clásico (ej: 20): ")
        MAX_ITER_GRASP = int(iter_grasp_str) if iter_grasp_str.isdigit() and int(iter_grasp_str) >0 else 20

        ALPHA_RCL_GRASP = 0.3
        RCL_SIZE_GREEDY = 3
        print(f"Config GRASP: WarmStarts={NUM_STOCH_WARM_STARTS}, MaxRestartsHC_WS={MAX_RESTARTS_HC_WARM_START}, GRASPIter={MAX_ITER_GRASP}, AlphaGRASP={ALPHA_RCL_GRASP}")

        plot_labels, plot_costs, plot_times, conv_hist_fi, conv_hist_bi = [], [], [], [], []

        # GRASP con HC Primera Mejora
        start_fi = time.time()
        layout_fi, cost_fi, conv_hist_fi = grasp_algorithm(
            num_planes,num_pists,E,P_array,L,Ci,Ck,tau,ALPHA_RCL_GRASP,MAX_ITER_GRASP,False,
            NUM_STOCH_WARM_STARTS, RCL_SIZE_GREEDY, MAX_RESTARTS_HC_WARM_START)
        

        time_fi = time.time() - start_fi

        sched_fi, feas_fi = (None,False)
        if layout_fi and cost_fi!=float('inf'): 
            _,sched_fi,feas_fi = ev_pist_layout(layout_fi,num_pists,E,P_array,L,Ci,Ck,tau)

        if not feas_fi: 
            cost_fi = float('inf')

        print("\nResultado GRASP + HC Alguna Mejora:"); print_runway_solution(sched_fi,cost_fi,feas_fi,layout_fi,num_pists)


        if feas_fi: 
            plot_labels.append("GRASP+HC Alguna Mejora")
            plot_costs.append(cost_fi); plot_times.append(time_fi)

        # GRASP con HC Mejor Mejora
        start_bi = time.time()
        layout_bi, cost_bi, conv_hist_bi = grasp_algorithm(
            num_planes,num_pists,E,P_array,L,Ci,Ck,tau,ALPHA_RCL_GRASP,MAX_ITER_GRASP,True,
            NUM_STOCH_WARM_STARTS, RCL_SIZE_GREEDY, MAX_RESTARTS_HC_WARM_START)
        
        time_bi = time.time() - start_bi

        sched_bi, feas_bi = (None,False)

        if layout_bi and cost_bi != float('inf'): 
            _,sched_bi,feas_bi = ev_pist_layout(layout_bi,num_pists,E,P_array,L,Ci,Ck,tau)
        if not feas_bi: cost_bi = float('inf')
        print("\nResultado GRASP + HC Mejor Mejora:") 
        print_runway_solution(sched_bi,cost_bi,feas_bi,layout_bi,num_pists)

        if feas_bi: 
            plot_labels.append("GRASP+HC Mejor Mejora")
            plot_costs.append(cost_bi)
            plot_times.append(time_bi)
        
        if plot_labels and plot_costs:
            plt.figure(figsize=(8,6)) 
            bars=plt.bar(plot_labels, plot_costs, color=['#1f77b4','#ff7f0e'])
            max_c=1

            if plot_costs:
                finite_c=[c for c in plot_costs if c!=float('inf')]

                if finite_c:
                    max_c=max(finite_c)

            for bar in bars:
                yval=bar.get_height()

                if yval!=float('inf'):
                    plt.text(bar.get_x() + bar.get_width() / 2., yval + 0.01 * max_c, f'{yval:.2f}', ha='center', va='bottom')

            plt.ylabel('Costo');plt.title(f'Comparación Costos GRASP\nCaso {select}, Pistas: {num_pists}') 
            plt.ylim(top=max_c*1.15 if max_c > 0 else 1.15)
            plt.tight_layout()
            plt.savefig(f"GRASP_costo_c{select}_p{num_pists}.png") 
            plt.show()


        if plot_labels and plot_times:
            plt.figure(figsize=(8,6))
            bars_t=plt.bar(plot_labels, plot_times, color=['#2ca02c', '#d62728'])
            max_t=1

            if plot_times:
                max_t=max(plot_times)

            for bar in bars_t:
                yval=bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., yval + 0.01 * max_t, f'{yval:.2f}s', ha='center', va='bottom')


            plt.ylabel('Tiempo (s)')
            plt.title(f'Comparación Tiempos GRASP\nCaso {select}, Pistas: {num_pists}')
            plt.ylim(top=max_t*1.15 if max_t > 0 else 1.15);plt.tight_layout()
            plt.savefig(f"GRASP_tiempo_c{select}_p{num_pists}.png");plt.show()

        plt.figure(figsize=(10,6))
        plotted_fi = False
        plotted_bi = False

        if conv_hist_fi and feas_fi:
            fi_pts=[(i,c) for i,c in enumerate(conv_hist_fi) if c!=float('inf')]

            if fi_pts:
                iters_fi, costs_fi=zip(*fi_pts)
                plt.plot(iters_fi, costs_fi, marker='o', ls='-', label='GRASP FI Conv.')
                plotted_fi=True

        if conv_hist_bi and feas_bi:
            bi_pts = [(i,c) for i,c in enumerate(conv_hist_bi) if c!=float('inf')]
            if bi_pts:iters_bi,costs_bi=zip(*bi_pts);plt.plot(iters_bi,costs_bi,marker='x',ls='--',label='GRASP BI Conv.') 
            plotted_bi = True

        if plotted_fi or plotted_bi:
            plt.xlabel('Punto de Muestreo del Mejor Costo');plt.ylabel('Mejor Costo Global')
            plt.title(f'Convergencia GRASP\nCaso {select}, Pistas: {num_pists}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"GRASP_conv_c{select}_p{num_pists}.png")
            plt.show()

    except FileNotFoundError: print(f"Error Crítico: No se encontró {filename}")
    except Exception as e: print(f"Error Crítico: {e}");

if __name__ == '__main__':
    main()
