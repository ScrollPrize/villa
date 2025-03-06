import os
import sys
# --------------------------------------------------
# Adjusted import for the custom graph problem library.
# --------------------------------------------------
sys.path.append('../ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

# --------------------------------------------------
# SolverInterface: Wraps interactions with the external solver library.
# --------------------------------------------------
class SolverInterface:
    def __init__(self, graph_path):
        self.graph_path = graph_path
        self.solver = graph_problem_gpu_py.Solver(graph_path)
    
    def load_graph(self, gt_path):
        self.solver.load_graph(gt_path)
    
    def get_positions(self):
        return self.solver.get_positions()
    
    def set_labels(self, labels, gt):
        self.solver.set_labels(labels, gt)
    
    def solve_winding_number(self, **kwargs):
        self.solver.solve_winding_number(**kwargs)
    
    def solve_union(self):
        self.solver.solve_union()
    
    def solve_random(self, **kwargs):
        self.solver.solve_random(**kwargs)
    
    def get_labels(self):
        return self.solver.get_labels()
    
    def save_solution(self, out_path):
        self.solver.save_solution(out_path)
