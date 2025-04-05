from copy import deepcopy
import time


def backJumping(coverage, costs):
    best_cost = float('inf')
    best_solution = None
    coms = list(costs.keys())
    

    coms_ord = sorted(coms, key=lambda i: -len(set(coverage[i])) / costs[i])
    
    def backtrack(not_covered, current_sol, current_cost, level):
        nonlocal best_cost, best_solution
        
        if not not_covered:
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = deepcopy(current_sol)
            return
        
        if level >= len(coms_ord):
            return
        
        commune = coms_ord[level]
        new_covered = set(coverage[commune]) & not_covered
        
        if new_covered:
            new_sol = current_sol.copy()
            new_sol[commune] = 1
            backtrack(not_covered - new_covered, new_sol, current_cost + costs[commune], level + 1)
        
        backtrack(not_covered, current_sol, current_cost, level + 1)
    
    # Ejecución
    start_time = time.time()
    backtrack(set(coverage.keys()), {}, 0, 0)
    execution_time = time.time() - start_time
    
    return best_solution, best_cost, execution_time


def greedy_deterministic(coverage, costs):
    comunas_ordenadas = sorted(coverage.keys(),
                             key=lambda c: -len(coverage[c])/costs[c])
    
    selected = []
    covered = set()
    total_cost = 0
    
    start_time = time.time()
    
    for comuna in comunas_ordenadas:
        # Verificar si esta comuna cubre áreas no cubiertas
        new_coverage = set(coverage[comuna]) - covered
        if new_coverage:
            selected.append(comuna)
            covered.update(coverage[comuna])
            total_cost += costs[comuna]
    
    execution_time = time.time() - start_time
    
    # Verificar si todas las comunas están cubiertas
    if len(covered) < len(coverage):
        return None, float('inf'), execution_time
    
    return selected, total_cost, execution_time


costs = {1: 60, 2: 30, 3: 60, 4: 70, 5: 130, 6: 60, 7: 70, 8: 60, 9: 80, 10: 70, 11: 50, 12: 90, 13: 30, 14: 30, 15: 100}

coverage = {
    1: [1, 2, 4, 13],
    2: [1, 2, 3, 4, 12, 15],
    3: [1, 3, 4, 5, 6, 13],
    4: [1, 2, 3, 4, 5, 12, 15],
    5: [3, 4, 5, 6, 7, 8, 12],
    6: [3, 5, 6, 9],
    7: [5, 7, 8, 10, 11, 12, 14, 15],
    8: [5, 7, 8, 9, 10],
    9: [6, 5, 8, 9, 10, 11],
    10: [7, 8, 9, 10, 11],
    11: [7, 9, 10, 11, 14],
    12: [2, 4, 5, 7, 12, 15],
    13: [1, 3, 13],
    14: [7, 11, 14],
    15: [2, 12, 7, 14]
}

print("\n=== Solución Backjumping Sin Heuristica ===\n")
bj_sol, bj_cost, bj_time = backJumping(coverage, costs)
if bj_sol:
    for comuna in sorted(bj_sol.keys()):
        if bj_sol[comuna] == 1:
            print(f"Construir en {comuna} (costo: {costs[comuna]})")
    print(f"\n-> Costo total: {bj_cost}")
    print(f"-> Tiempo de ejecución: {bj_time:.4f} segundos")
else:
    print("No se encontró solución")


print("=== Solución Greedy Determinista ===\n")
greedy_sol, greedy_cost, greedy_time = greedy_deterministic(coverage, costs)
if greedy_sol:
    for comuna in greedy_sol:
        print(f"Construir en {comuna} (costo: {costs[comuna]})")
    print(f"\n-> Costo total: {greedy_cost}")
    print(f"-> Tiempo de ejecución: {greedy_time:.4f} segundos")
else:
    print("No se encontró solución válida")


#comparacion
print("\n=============================================\n")
print(f"Diferencia de costo (Greedy - Backjumping): {greedy_cost - bj_cost}")
print(f"Tiempo Greedy: {greedy_time:.4f}seg vs Backjumping: {bj_time:.4f}seg")
print(f"Relación de tiempo: {greedy_time/bj_time:.4f}seg más rápido")