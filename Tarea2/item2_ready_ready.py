
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
seq_pista = List[PlaneIdx]
MultiPist = Dict[PistIdx, seq_pista] 
OverallSchedule = Dict[PlaneIdx, int] 


def read_file(path):
    try:
        with open(path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        sys.exit(1)

    line_idx = 0
    try:
        if line_idx >= len(lines): raise ValueError("File seems empty.")
        num_planes = int(lines[line_idx])
        line_idx += 1
        E = []
        P = []
        L = []
        Ci = []
        Ck = []
        tau = []


        for i in range(num_planes):

            if line_idx >= len(lines): raise ValueError(f"EOF for plane {i} data.")
            parts = lines[line_idx].split(); line_idx += 1

            if len(parts) != 5: raise ValueError(f"Plane {i} data: Expected 5 values, got {len(parts)}")

            E.append(int(parts[0])); P.append(int(parts[1])); L.append(int(parts[2]))
            Ci.append(float(parts[3])); Ck.append(float(parts[4]))
            
            current_sep_row = []
            while len(current_sep_row) < num_planes:

                if line_idx >= len(lines): raise ValueError(f"EOF for plane {i} sep times.")

                sep_parts = lines[line_idx].split(); line_idx += 1
                current_sep_row.extend([int(p) for p in sep_parts])

            if len(current_sep_row) != num_planes: raise ValueError(f"Plane {i} sep times: Expected {num_planes}, got {len(current_sep_row)}")
            tau.append(current_sep_row)

        if len(E) != num_planes: raise ValueError("Plane data count mismatch.")

        E,P,L,Ci,Ck,tau = np.array(E), np.array(P), np.array(L), np.array(Ci), np.array(Ck), np.array(tau)


        if not (np.all(E <= P) and np.all(P <= L)): print("Warning: E <= P <= L constraint violated.")

        return num_planes, E, P, L, Ci, Ck, tau
    except Exception as e:
        print(f"Error reading file '{path}': {e}"); sys.exit(1)


def total_cost(overall_schedule: OverallSchedule, P: np.ndarray, Ci: np.ndarray, Ck: np.ndarray) -> float:
    if not overall_schedule: return float('inf')
    total_cost = 0.0
    for plane_idx, landing_time in overall_schedule.items():
        if plane_idx < 0 or plane_idx >= len(P):
            print(f"Cost Error: Invalid plane_idx {plane_idx}")
            return float('inf')
        penalty = Ci[plane_idx] * max(0, P[plane_idx] - landing_time) + \
                  Ck[plane_idx] * max(0, landing_time - P[plane_idx])
        total_cost += penalty
    return total_cost


def multi_pist_conf(
    multi_runway_layout: MultiPist,
    num_runways: int,
    E: np.ndarray, L: np.ndarray, tau: np.ndarray
) -> Tuple[Optional[OverallSchedule], bool]:

    overall_schedule: OverallSchedule = {}
    last_landing_time_on_runway: Dict[PistIdx, int] = {r: -1 for r in range(num_runways)}
    last_plane_on_runway: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_runways)}
    
    all_planes_in_layout = set()
    for runway_idx in range(num_runways):
        if runway_idx not in multi_runway_layout: #si esta bien hecho no pasa
            multi_runway_layout[runway_idx] = [] 

        for plane_idx in multi_runway_layout[runway_idx]:
            all_planes_in_layout.add(plane_idx)
            min_start_time = E[plane_idx]
            earliest_after_separation = 0

            if last_plane_on_runway[runway_idx] != -1:
                prev_plane = last_plane_on_runway[runway_idx]
                separation_needed = tau[prev_plane][plane_idx]
                if separation_needed >= 99999: return None, False 
                earliest_after_separation = last_landing_time_on_runway[runway_idx] + separation_needed
            
            actual_landing_time = max(min_start_time, earliest_after_separation)

            if actual_landing_time > L[plane_idx]: return None, False # Violates latest landing time

            overall_schedule[plane_idx] = actual_landing_time
            last_landing_time_on_runway[runway_idx] = actual_landing_time
            last_plane_on_runway[runway_idx] = plane_idx
    
    # Check if all planes were scheduled 
    num_total_planes = len(E)
    if len(all_planes_in_layout) != num_total_planes:
        pass

    return overall_schedule, True


def ev_multi_pist_layout(
    multi_runway_layout: MultiPist, num_runways: int,
    E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray
) -> Tuple[float, Optional[OverallSchedule], bool]:
    """
    Evaluates a multi-runway layout: calculates schedule, checks feasibility, computes cost.
    """
    overall_schedule, feasible = multi_pist_conf(multi_runway_layout, num_runways, E, L, tau)

    if not feasible or overall_schedule is None:
        return float('inf'), None, False
    
  
    if len(overall_schedule) != len(E):
        return float('inf'), None, False # cronograma no factible

    cost = total_cost(overall_schedule, P, Ci, Ck)
    return cost, overall_schedule, True

def greedy_deterministic_multi(
    num_planes: int, num_pist: int, E: np.ndarray, P_array: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray) -> Tuple[Optional[MultiPist], float, bool]:

    planes_sorted_by_P = sorted(range(num_planes), key=lambda k: (P_array[k], E[k]))

    solution_layout: MultiPist = {r: [] for r in range(num_pist)}
    last_landing_time_on_pist: Dict[PistIdx, int] = {r: -1 for r in range(num_pist)}
    lastP_pist: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_pist)}
    
    scheduled_planes_count = 0

    for plane_to_time_idx in planes_sorted_by_P:

        best_pist = -1
        e_landing_for_plane = float('inf')

        for r_idx in range(num_pist):
            min_start_time = E[plane_to_time_idx]
            ear_after_sep_pist = 0

            if lastP_pist[r_idx] != -1:
                prev_plane = lastP_pist[r_idx]
                sep = tau[prev_plane][plane_to_time_idx]

                if sep >= 99999: continue 
                ear_after_sep_pist = last_landing_time_on_pist[r_idx] + sep
            
            current_pTime = max(min_start_time, ear_after_sep_pist)

            if current_pTime <= L[plane_to_time_idx]:

                if current_pTime < e_landing_for_plane:
                    e_landing_for_plane = current_pTime
                    best_pist = r_idx
        
        if best_pist == -1: 
            return None, float('inf'), False 

        # asignacion mejor pista
        solution_layout[best_pist].append(plane_to_time_idx)
        last_landing_time_on_pist[best_pist] = e_landing_for_plane
        lastP_pist[best_pist] = plane_to_time_idx
        scheduled_planes_count +=1

    if scheduled_planes_count != num_planes:
        return None, float('inf'), False # no todos los aviones

    cost, _, feasible = ev_multi_pist_layout(solution_layout, num_pist, E, P_array, L, Ci, Ck, tau)
    return solution_layout, cost, feasible


def greedy_stochastic_multi(num_planes: int, num_runways: int, E: np.ndarray, P_array: np.ndarray, L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, rcl_size: int = 3, seed: Optional[int] = None
    ) -> Tuple[Optional[MultiPist], float, bool]:

    if seed is not None: random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    solution_layout: MultiPist = {r: [] for r in range(num_runways)}
    last_landing_time_on_runway: Dict[PistIdx, int] = {r: -1 for r in range(num_runways)}
    last_plane_on_runway: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_runways)}

    while unscheduled_planes:
        candidates = [] 
        for k_plane in unscheduled_planes:
            for r_idx in range(num_runways):

                min_start_time = E[k_plane]
                earliest_after_sep = 0

                if last_plane_on_runway[r_idx] != -1:
                    prev_p = last_plane_on_runway[r_idx]
                    sep = tau[prev_p][k_plane]

                    if sep >= 99999: continue
                    earliest_after_sep = last_landing_time_on_runway[r_idx] + sep
                
                possible_time = max(min_start_time, earliest_after_sep)

                if possible_time <= L[k_plane]:
                    candidates.append({'plane': k_plane, 'runway': r_idx, 'time': possible_time})
        
        if not candidates: return None, float('inf'), False 

        candidates.sort(key=lambda x: x['time']) # de menor a mayor
        current_rcl_limit = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_limit]
        
        chosen_candidate = random.choice(rcl)
        chosen_plane = chosen_candidate['plane']
        chosen_runway = chosen_candidate['runway']
        chosen_time = chosen_candidate['time']

        solution_layout[chosen_runway].append(chosen_plane)
        last_landing_time_on_runway[chosen_runway] = chosen_time
        last_plane_on_runway[chosen_runway] = chosen_plane
        unscheduled_planes.remove(chosen_plane)

    if len(unscheduled_planes) > 0: return None, float('inf'), False

    cost, _, feasible = ev_multi_pist_layout(solution_layout, num_runways, E, P_array, L, Ci, Ck, tau)
    return solution_layout, cost, feasible



def construct_greedy_randomized_multi(
    num_planes: int, num_runways: int, E: np.ndarray, L: np.ndarray, tau: np.ndarray, alpha: float
) -> Optional[MultiPist]:
    unscheduled_planes = set(range(num_planes))
    solution_layout: MultiPist = {r: [] for r in range(num_runways)}
    last_landing_time_on_runway: Dict[PistIdx, int] = {r: -1 for r in range(num_runways)}
    last_plane_on_runway: Dict[PistIdx, PlaneIdx] = {r: -1 for r in range(num_runways)}

    for _ in range(num_planes):
        candidates = [] 
        min_cand_time, max_cand_time = float('inf'), -float('inf')

        for k_plane in unscheduled_planes:
            for r_idx in range(num_runways):
                min_start = E[k_plane]
                earliest_sep = 0


                if last_plane_on_runway[r_idx] != -1:
                    prev_p = last_plane_on_runway[r_idx]
                    sep = tau[prev_p][k_plane]


                    if sep >= 99999: continue
                    earliest_sep = last_landing_time_on_runway[r_idx] + sep
                
                p_time = max(min_start, earliest_sep)


                if p_time <= L[k_plane]:
                    candidates.append({'plane': k_plane, 'runway': r_idx, 'time': p_time})
                    min_cand_time = min(min_cand_time, p_time)
                    max_cand_time = max(max_cand_time, p_time)
        
        if not candidates: return None # Failed construction

        threshold = min_cand_time + alpha * (max_cand_time - min_cand_time) if min_cand_time != max_cand_time else min_cand_time
        rcl = [c for c in candidates if c['time'] <= threshold]

        #restricciones
        if not rcl: rcl = [c for c in candidates if c['time'] == min_cand_time] # Fallback
        if not rcl: return None 

        chosen_cand = random.choice(rcl)
        c_plane, c_runway, c_time = chosen_cand['plane'], chosen_cand['runway'], chosen_cand['time']

        solution_layout[c_runway].append(c_plane)
        last_landing_time_on_runway[c_runway] = c_time
        last_plane_on_runway[c_runway] = c_plane
        unscheduled_planes.remove(c_plane)
        
    if len(unscheduled_planes) > 0: return None
    return solution_layout


def hill_climbing_multi(
    initial_layout: MultiPist, num_pist: int,
    E: np.ndarray, P_array: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    use_best_improvement: bool) -> Tuple[MultiPist, float]:


    current_layout = copy.deepcopy(initial_layout)
    current_cost, _, feasible = ev_multi_pist_layout(current_layout, num_pist, E, P_array, L, Ci, Ck, tau)


    if not feasible: return initial_layout, float('inf')


    while True:
        best_neighbor_layout = None
        best_neighbor_cost = current_cost
        moved_in_iter = False

        # Intercambios en la pista
        for r_idx in range(num_pist):
            pista_seq = current_layout[r_idx]
            if len(pista_seq) < 2: continue

            for i in range(len(pista_seq)):
                for j in range(i + 1, len(pista_seq)):

                    neighbor_layout = copy.deepcopy(current_layout)
                    neighbor_layout[r_idx][i], neighbor_layout[r_idx][j] = neighbor_layout[r_idx][j], neighbor_layout[r_idx][i]
                    
                    cost_n, _, feas_n = ev_multi_pist_layout(neighbor_layout, num_pist, E, P_array, L, Ci, Ck, tau)


                    if feas_n and cost_n < best_neighbor_cost:
                        if use_best_improvement:
                            best_neighbor_cost, best_neighbor_layout = cost_n, neighbor_layout

                        else: 
                            current_layout, current_cost = neighbor_layout, cost_n
                            moved_in_iter = True; break 
                        
                if not use_best_improvement and moved_in_iter: break
            if not use_best_improvement and moved_in_iter: break
        if not use_best_improvement and moved_in_iter:
            continue 

        # movimientos entre pistas
        if num_pist > 1: 
            for src_r_idx in range(num_pist):
                if not current_layout[src_r_idx]: continue 
                for plane_pos_in_src in range(len(current_layout[src_r_idx])):
                    plane_to_move = current_layout[src_r_idx][plane_pos_in_src]
                    for dest_r_idx in range(num_pist):
                        if src_r_idx == dest_r_idx: continue

                        neighbor_layout = copy.deepcopy(current_layout)

                        moved_plane = neighbor_layout[src_r_idx].pop(plane_pos_in_src)
                        neighbor_layout[dest_r_idx].append(moved_plane)
                        
                        cost_n, _, feas_n = ev_multi_pist_layout(neighbor_layout, num_pist, E, P_array, L, Ci, Ck, tau)
                        if feas_n and cost_n < best_neighbor_cost:
                            if use_best_improvement:
                                best_neighbor_cost, best_neighbor_layout = cost_n, neighbor_layout
                            else:
                                current_layout, current_cost = neighbor_layout, cost_n
                                moved_in_iter = True; break
                    if not use_best_improvement and moved_in_iter: break
                if not use_best_improvement and moved_in_iter: break
            if not use_best_improvement and moved_in_iter:
                continue # Restart while loop

        if use_best_improvement and best_neighbor_layout is not None and best_neighbor_cost < current_cost:
            current_layout, current_cost = best_neighbor_layout, best_neighbor_cost

        elif not moved_in_iter and (not use_best_improvement or (use_best_improvement and best_neighbor_cost >= current_cost)):
             break 

    return current_layout, current_cost

""" def grasp_multi(
    num_planes: int, num_pist: int, E: np.ndarray, P_array: np.ndarray, 
    L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    alpha: float, max_it: int, use_mejor_mejora: bool,
    initial_layouts: List[MultiPist]) -> Tuple[Optional[MultiPist], float, List[float]]:

    best_overall_layout: Optional[MultiPist] = None
    best_overall_cost: float = float('inf')
    cost_history: List[float] = []
    start_time_grasp = time.time()

    print(f"\n--- Starting GRASP (Pistas: {num_pist}) ---")
    print(f"Params: alpha={alpha}, max_iter={max_it}, HC={'BI' if use_mejor_mejora else 'FI'}")

    # Phase 1: Improve Initial Layouts
    valid_initial_layouts = [lyt for lyt in initial_layouts if lyt is not None]
    print(f"\n[GRASP] Phase 1: Improving {len(valid_initial_layouts)} initial layout(s)...")
    for i, init_layout in enumerate(valid_initial_layouts):
        hc_layout, hc_cost = hill_climbing_multi(init_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora)
        if hc_cost < best_overall_cost:
            best_overall_cost, best_overall_layout = hc_cost, hc_layout
            print(f"    New best from initial HC #{i+1}: {hc_cost:.2f}")

    cost_str = f"{best_overall_cost:.2f}" if best_overall_cost != float('inf') else "inf"
    print(f"[GRASP] Best cost after Phase 1: {cost_str}")

    

    # Phase 2: GRASP Iterations
    print(f"\n[GRASP] Phase 2: Starting {max_it} construction/local search iterations...")
    for i in range(max_it):
        constructed_layout = construct_greedy_randomized_multi(num_planes, num_pist, E, L, tau, alpha)
        if constructed_layout is None: 
            if best_overall_cost != float('inf'): 
                cost_history.append(best_overall_cost)
            continue

        improved_layout, improved_cost = hill_climbing_multi(constructed_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora)


        if improved_cost < best_overall_cost:
            best_overall_cost, best_overall_layout = improved_cost, improved_layout
            print(f"    Iter {i+1}: New best solution! Cost: {improved_cost:.2f}")

        if best_overall_cost != float('inf'):
            cost_history.append(best_overall_cost)

        if (i + 1) % 10 == 0: print(f"  GRASP Iter {i+1}/{max_it} done. Best: {best_overall_cost:.2f}")
    
    print(f"\n[GRASP] Finished in {time.time() - start_time_grasp:.4f}s.")
    if best_overall_layout is None:
        print("[GRASP] No feasible solution found.")
        return None, float('inf')
    
    final_cost, _, _ = ev_multi_pist_layout(best_overall_layout, num_pist, E, P_array, L, Ci, Ck, tau)
    print(f"[GRASP] Best solution cost verified: {final_cost:.2f}")

   
    return best_overall_layout, final_cost """

# ... (código anterior de grasp_multi) ...

def grasp_multi(
    num_planes: int, num_pist: int, E: np.ndarray, P_array: np.ndarray,
    L: np.ndarray,
    Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray,
    alpha: float, max_it: int, use_mejor_mejora: bool,
    initial_layouts: List[MultiPist]
) -> Tuple[Optional[MultiPist], float, List[float]]: # <--- Añadido List[float] para el historial

    best_overall_layout: Optional[MultiPist] = None
    best_overall_cost: float = float('inf')
    cost_history: List[float] = [] # <--- Lista para guardar el historial de costos
    start_time_grasp = time.time()

    print(f"\n--- Starting GRASP (Pistas: {num_pist}) ---")
    print(f"Params: alpha={alpha}, max_iter={max_it}, HC={'BI' if use_mejor_mejora else 'FI'}")

    # Phase 1: Improve Initial Layouts
    valid_initial_layouts = [lyt for lyt in initial_layouts if lyt is not None]
    # print(f"\n[GRASP] Phase 1: Improving {len(valid_initial_layouts)} initial layout(s)...") # Más verboso
    initial_improvement_made = False
    for i, init_layout in enumerate(valid_initial_layouts):
        hc_layout, hc_cost = hill_climbing_multi(init_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora)
        if hc_cost < best_overall_cost:
            best_overall_cost, best_overall_layout = hc_cost, hc_layout
            # print(f"    New best from initial HC #{i+1}: {hc_cost:.2f}") # Más verboso
            initial_improvement_made = True
    
    # Registrar el costo después de la fase de inicialización si hubo alguna mejora o si se encontró alguna solución factible
    if best_overall_cost != float('inf'):
        cost_history.append(best_overall_cost) # Primer punto del historial
        cost_str_phase1 = f"{best_overall_cost:.2f}"
    else:
        cost_str_phase1 = "inf"
        # Si no hay solución factible inicial, el historial podría empezar con inf o esperar a la primera solución de GRASP
        # Por simplicidad, lo dejaremos así y el gráfico manejará los 'inf' (o los ignorará).

    if initial_improvement_made:
        print(f"[GRASP] Best cost after Phase 1 (initial HC): {cost_str_phase1}")
    elif len(valid_initial_layouts) > 0:
         print(f"[GRASP] No improvement from initial solutions. Current best (if any from initial): {cost_str_phase1}")
    else:
        print(f"[GRASP] No valid initial layouts provided for Phase 1.")


    # Phase 2: GRASP Iterations
    # print(f"\n[GRASP] Phase 2: Starting {max_it} construction/local search iterations...") # Más verboso
    for i in range(max_it):
        constructed_layout = construct_greedy_randomized_multi(num_planes, num_pist, E, L, tau, alpha)
        
        current_iter_cost = float('inf') # Costo de esta iteración particular

        if constructed_layout is not None:
            improved_layout, improved_cost_iter = hill_climbing_multi(constructed_layout, num_pist, E, P_array, L, Ci, Ck, tau, use_mejor_mejora)
            current_iter_cost = improved_cost_iter

            if improved_cost_iter < best_overall_cost:
                best_overall_cost, best_overall_layout = improved_cost_iter, improved_layout
                # print(f"    Iter {i+1}: New best solution! Cost: {improved_cost_iter:.2f}") # Más verboso
        
        # Registrar el *mejor costo global encontrado hasta esta iteración*
        if best_overall_cost != float('inf'):
            cost_history.append(best_overall_cost)
        elif cost_history: # Si ya hay algo en el historial (ej. de la fase inicial), repetimos el último valor si no mejora
            cost_history.append(cost_history[-1])
        # Si best_overall_cost sigue siendo inf y cost_history está vacío, no agregamos inf repetidamente aquí.
        # El gráfico puede empezar desde la primera vez que se encuentra una solución factible.

        if (i + 1) % 10 == 0 or i == max_it -1 : # Imprimir progreso
             prog_cost_str = f"{best_overall_cost:.2f}" if best_overall_cost != float('inf') else "inf"
             print(f"  GRASP Iter {i+1}/{max_it} done. Current best global cost: {prog_cost_str}")
    
    # print(f"\n[GRASP] Finished in {time.time() - start_time_grasp:.4f}s.") # Más verboso
    if best_overall_layout is None:
        print("[GRASP] No feasible solution found after all iterations.")
        # Si no se encontró solución, el historial podría estar vacío o lleno de 'inf'
        # Asegurémonos de que el historial no esté completamente vacío si se va a graficar
        if not cost_history and max_it > 0:
            cost_history = [float('inf')] * max_it # Llenar con inf si no se encontró nada
        return None, float('inf'), cost_history # <--- Devolver historial

    # Verificar costo final (opcional, ya debería ser best_overall_cost)
    # final_cost_check, _, _ = ev_multi_pist_layout(best_overall_layout, num_pist, E, P_array, L, Ci, Ck, tau)
    # print(f"[GRASP] Best solution cost (from tracking): {best_overall_cost:.2f}, Verified final: {final_cost_check:.2f}")
    
    return best_overall_layout, best_overall_cost, cost_history # <--- Devolver historial


def print_solution_multi(
    overall_schedule: Optional[OverallSchedule],
    total_cost: float,
    feasible: bool,
    mult_pist_layout: Optional[MultiPist] = None,
    num_pist: Optional[int] = None
):
    cost_str = "inf" if total_cost == float('inf') else f"{total_cost:.2f}"
    if not feasible or overall_schedule is None:
        print(f"No se encontró solución factible. Costo: {cost_str}")
        if mult_pist_layout:
            print("Layout (parcial/infactible):")
            for r, planes in mult_pist_layout.items(): print(f"  Pista {r+1}: {planes}")
        return

    print(f"Costo total: {total_cost:.2f}")
    if mult_pist_layout and num_pist is not None:
        print("\nTiempo de aterrizaje por pista:")
        print("\nAvion @ tiempo\n")
        det_planes_pists = {r: [] for r in range(num_pist)}

        sorted_planes_by_time = sorted(overall_schedule.items(), key=lambda item: item[1])
        
        map_plane_pist: Dict[PlaneIdx, PistIdx] = {}
        for r_idx, planes_in_r in mult_pist_layout.items():
            for p_idx in planes_in_r:
                map_plane_pist[p_idx] = r_idx

        for p_idx, landing_time in sorted_planes_by_time:

            if p_idx in map_plane_pist:
                 r_assigned = map_plane_pist[p_idx]
                 det_planes_pists[r_assigned].append(f"Avión {p_idx+1}@{landing_time}")


            else:
                 print(f"Warning: Plane {p_idx+1} in schedule but not in layout map.")


        for r_idx in range(num_pist):
            print(f"\n  Pista {r_idx+1}: \n" + ", ".join(det_planes_pists[r_idx]))

    print("\nHorario General (Avión: Tiempo):")
    print("Avión\tTiempo de aterrizaje")
    print("-" * 25)

    sorted_schedule_items = sorted(overall_schedule.items(), key=lambda item: item[1])
    for plane_idx, landing_time in sorted_schedule_items:
        print(f"{plane_idx + 1}\t{landing_time}")

def run_multiple_stochastic_multi(
    num_planes, num_runways, E, P_array, L, Ci, Ck, tau, num_runs=10, rcl_size=3) -> List[Dict[str, Any]]:


    results = []
    for i in range(num_runs):
        seed = i
        start_time = time.time()
        layout, cost, feasible = greedy_stochastic_multi(num_planes, num_runways, E, P_array, L, Ci, Ck, tau, rcl_size, seed)
        end_time = time.time()
        
        overall_schedule = None

        if feasible and layout:
            cost_check, schedule_check, feasible_check = ev_multi_pist_layout(layout, num_runways, E, P_array, L, Ci, Ck, tau)

            if feasible_check and schedule_check:
                overall_schedule = schedule_check

                if abs(cost - cost_check) > 1e-6: cost = cost_check

            else: feasible, cost, layout = False, float('inf'), None


        results.append({"seed": seed, "cost": cost, "time": end_time-start_time, "layout": layout, "schedule": overall_schedule, "feasible": feasible})
    
    return results

def main():
    DEFAULT_CASE_DIR = "casos"
    select_str = input("Selecciona el caso (1-4) o 0 para salir: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n")
    
    
    try:
        select = int(select_str)
        if select == 0: print("Saliendo..."); sys.exit()
        if not 1 <= select <= 4: print("Opción no válida."); sys.exit()
    except ValueError: print("Entrada inválida."); sys.exit()

    while True:
        try:
            num_runways_str = input("Ingrese el número de pistas a utilizar (1 o 2): ")
            num_pist = int(num_runways_str)

            if num_pist not in [1, 2]:
                print("Número de pistas debe ser 1 o 2.")

            else:
                break
        except ValueError:

            print("Por favor ingrese un número entero (1 o 2).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir if script_dir else os.getcwd()
    filename = os.path.join(base_dir, DEFAULT_CASE_DIR, f'case{select}.txt')
    print(f"\nLeyendo instancia desde: {filename}")

    try:
        num_planes, E, P, L, Ci, Ck, tau = read_file(filename) 

        print("-" * 40 + f"\nLectura exitosa. Aviones: {num_planes}, Pistas: {num_pist}" + "\n" + "-" * 40)

        print("\n" + "=" * 40 + "\nALGORITMOS GREEDY (Puntos de partida GRASP)\n" + "=" * 40)

        print("\n--- Greedy Determinista (Multi-Pista) ---")

        start_det = time.time()
        det_layout, det_cost, det_feasible = greedy_deterministic_multi(num_planes, num_pist, E, P, L, Ci, Ck, tau)
        print(f"Tiempo: {time.time() - start_det:.4f}s")
        det_schedule = None


        if det_feasible and det_layout:
            _, det_schedule, _ = ev_multi_pist_layout(det_layout, num_pist, E, P, L, Ci, Ck, tau)
        print_solution_multi(det_schedule, det_cost, det_feasible, det_layout, num_pist)

        print("\n--- Greedy Estocástico (Multi-Pista, 10 Ejecuciones) ---")
        stochastic_results = run_multiple_stochastic_multi(num_planes, num_pist, E, P, L, Ci, Ck, tau)
        
        feasible_stochastic_layouts: List[MultiPist] = []
        best_sto_cost, best_sto_result = float('inf'), None
        print("\nResumen Estocástico:")
        print("Seed\tCosto\t\tTiempo (s)\tFactible")
        print("-" * 50)


        for res in stochastic_results:
            cost_disp = f"{res['cost']:.2f}" if res['feasible'] else "inf"
            print(f"{res['seed']}\t{cost_disp}\t\t{res['time']:.4f}\t\t{res['feasible']}")


            if res['feasible'] and res['layout']:
                feasible_stochastic_layouts.append(res['layout'])

                if res['cost'] < best_sto_cost: best_sto_cost, best_sto_result = res['cost'], res

        print("-" * 50)


        if best_sto_result:
            print("\nMejor solución estocástica:")
            print_solution_multi(best_sto_result['schedule'], best_sto_result['cost'], True, best_sto_result['layout'], num_pist)

        
        plot_labels = []
        plot_costs = []
        convergence_histories = {}


        initial_grasp_layouts = []

        if det_feasible and det_layout: initial_grasp_layouts.append(det_layout)

        initial_grasp_layouts.extend(feasible_stochastic_layouts)
 
        print(f"\nLayouts iniciales válidos para GRASP: {len(initial_grasp_layouts)}")

        print("\n" + "=" * 40 + "\nALGORITMO GRASP (Multi-Pista)\n" + "=" * 40)
        GRASP_ALPHA, GRASP_MAX_ITERATIONS = 0.3, 50 # Tune these



        print("\n--- GRASP con Hill Climbing: ALGUNA-MEJORA ---")


        grasp_fi_layout, grasp_fi_cost, history_fi = grasp_multi(
            num_planes, num_pist, E, P, L, Ci, Ck, tau,
            GRASP_ALPHA, GRASP_MAX_ITERATIONS, False, copy.deepcopy(initial_grasp_layouts))
        
        grasp_fi_schedule, grasp_fi_feasible = (None, False)


        if grasp_fi_layout and grasp_fi_cost != float('inf'):
            _, grasp_fi_schedule, grasp_fi_feasible = ev_multi_pist_layout(grasp_fi_layout, num_pist, E, P, L, Ci, Ck, tau)

        label_fi = f"GRASP + HC Primera Mejora (Pistas: {num_pist})"
        if grasp_fi_feasible:
            convergence_histories[label_fi] = history_fi
        else: 
            convergence_histories[label_fi] = [float('inf')] * len(history_fi) if history_fi else [float('inf')] * GRASP_MAX_ITERATIONS


        print("\nMejor solución GRASP (ALGUNA-MEJORA):")
        print_solution_multi(grasp_fi_schedule, grasp_fi_cost, grasp_fi_feasible, grasp_fi_layout, num_pist)
        if grasp_fi_feasible: 
            plot_labels.append(f"GRASP + HC Primera Mejora\n(Pistas: {num_pist})")

            plot_costs.append(grasp_fi_cost)

            #---------------------------------------------

        print("\n--- GRASP con Hill Climbing: MEJOR-MEJORA ---")

        grasp_bi_layout, grasp_bi_cost, history_bi = grasp_multi(
            num_planes, num_pist, E, P, L, Ci, Ck, tau,
            GRASP_ALPHA, GRASP_MAX_ITERATIONS, True, copy.deepcopy(initial_grasp_layouts))
        grasp_bi_schedule, grasp_bi_feasible = (None, False)
        if grasp_bi_layout and grasp_bi_cost != float('inf'):
            _, grasp_bi_schedule, grasp_bi_feasible = ev_multi_pist_layout(grasp_bi_layout, num_pist, E, P, L, Ci, Ck, tau)

        label_bi = f"GRASP + HC Mejor Mejora (Pistas: {num_pist})"
        if grasp_bi_feasible:
            convergence_histories[label_bi] = history_bi
        else:
            convergence_histories[label_bi] = [float('inf')] * len(history_bi) if history_bi else [float('inf')] * GRASP_MAX_ITERATIONS

        print("\nMejor solución GRASP (Mejor Mejora):")
        print_solution_multi(grasp_bi_schedule, grasp_bi_cost, grasp_bi_feasible, grasp_bi_layout, num_pist)

        if grasp_bi_feasible: 
            plot_labels.append(f"GRASP + HC Mejor Mejora\n(Pistas: {num_pist})")
            plot_costs.append(grasp_bi_cost)

        print("\n" + "=" * 40 + "\nCOMPARACIÓN FINAL GRASP\n" + "=" * 40)

        fi_str = f"{grasp_fi_cost:.2f}" if grasp_fi_feasible else "inf"
        bi_str = f"{grasp_bi_cost:.2f}" if grasp_bi_feasible else "inf"


        print(f"Costo GRASP + HC Alguna-Mejora : {fi_str}")
        print(f"Costo GRASP + HC Mejor-Mejora   : {bi_str}")
        print("=" * 40)

        if plot_labels and plot_costs:
           plt.figure(figsize=(10, 6)) 
           bars = plt.bar(plot_labels, plot_costs, color=['skyblue', 'lightcoral'])

         
           for bar in bars:
               yval = bar.get_height()
               if yval != float('inf'):
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(plot_costs), f'{yval:.2f}', ha='center', va='bottom')


           plt.ylabel('Costo Total de Penalización')
           plt.title(f'Comparación de Costos: GRASP con Hill Climbing\nCaso {select}, Pistas: {num_pist}')
           plt.xticks(rotation=0) 
           plt.tight_layout() 
           
        
           plot_filename = f"comparacion_grasp_caso{select}_pistas{num_pist}.png"
           plt.savefig(plot_filename)
           print(f"\nGráfico guardado como: {plot_filename}")
           
           plt.show() 
        else:
            print("\nNo se generó gráfico porque no se obtuvieron resultados factibles para comparar.")
        

        if convergence_histories:
            plt.figure(figsize=(12, 7))
            for label, history in convergence_histories.items():
                # Filtrar valores inf para que no arruinen la escala del gráfico,
                # o reemplazarlos por un valor muy alto si se quiere ver que no convergieron a algo bueno.
                # Opción 1: Filtrar 'inf'
                iterations = [i for i, cost in enumerate(history) if cost != float('inf')]
                costs_to_plot = [cost for cost in history if cost != float('inf')]

                # Opción 2: Reemplazar 'inf' (puede ser útil si quieres ver un plato)
                # max_sensible_cost = 0
                # for h in convergence_histories.values():
                #     for c_val in h:
                #         if c_val != float('inf') and c_val > max_sensible_cost:
                #             max_sensible_cost = c_val
                # if max_sensible_cost == 0: max_sensible_cost = 1 # Evitar 0 si todo fue inf

                # costs_to_plot = [c if c != float('inf') else max_sensible_cost * 1.1 for c in history] # Reemplazar inf
                # iterations = list(range(len(costs_to_plot)))

                if costs_to_plot: # Solo graficar si hay puntos válidos
                    plt.plot(iterations, costs_to_plot, marker='o', linestyle='-', markersize=4, label=label)
                else:
                    print(f"No se graficó la convergencia para '{label}' porque no se encontraron costos factibles en su historial.")


            plt.xlabel("Iteración de GRASP (o punto de registro)")
            plt.ylabel("Mejor Costo Encontrado Hasta el Momento")
            plt.title(f"Convergencia de GRASP: Caso {select}, Pistas: {num_pist}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            convergence_plot_filename = f"convergencia_grasp_caso{select}_pistas{num_pist}.png"
            plt.savefig(convergence_plot_filename)
            print(f"\nGráfico de convergencia guardado como: {convergence_plot_filename}")
            plt.show()
    
        else:
            print("\nNo se generó gráfico de convergencia porque no hay historiales para mostrar.")
    
    except Exception as e:
        print(f"Error Crítico: {e}")
        import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
    