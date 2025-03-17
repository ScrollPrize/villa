import numpy as np
import sys
import cv2
import os
import argparse

sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

def compress_video(input_path, output_path, codec='mp4v', scale=0.5):
    """
    Compress a video by resizing its frames and re-encoding it.
    
    :param input_path: Path to the input video.
    :param output_path: Path to save the compressed video.
    :param codec: FourCC code for the codec (default: 'mp4v' for MP4 files).
    :param scale: Scaling factor for the video frames (default: 0.5 for half the size).
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original Video: {original_width}x{original_height}, FPS: {original_fps}, Frames: {frame_count}")
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    print(f"Compressed Video Dimensions: {new_width}x{new_height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (new_width, new_height))
    
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Write the resized frame to the output video
        out.write(resized_frame)
        
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{frame_count} frames...")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Compressed video saved to {output_path}")

def find_seed_node(solver):
    # solver.load_graph("checkpoint_graph_2.bin")
    points = np.array(solver.get_positions())
    min_f_star_percentile = 0.01
    max_f_star_percentile = 0.99
    min_f_star = np.percentile(points[:,0], min_f_star_percentile)
    max_f_star = np.percentile(points[:,0], max_f_star_percentile)
    middle_f_star = (min_f_star + max_f_star + 4010) / 2
    print("middle_f_star", middle_f_star)
    middle_f_star = 500
    min_z = np.min(points[:,2])
    max_z = np.max(points[:,2])
    search_point = np.array([middle_f_star, 140, (min_z + max_z) / 2])
    search_point = np.array([500, 140, 2875]) # fixed inside a good nice region
    print("search_point", search_point)
    # find closest index to search point
    distances = np.linalg.norm(points - search_point, axis=1)
    closest_index = np.argmin(distances)
    print("closest_index", closest_index)
    print("closest distance", distances[closest_index])
    return closest_index

def update_sides(solver):
    sides = np.array(solver.get_sides())
    side_number_additions = sides > 0
    side_numbers = np.array(solver.get_side_numbers())
    side_numbers = 2 * side_numbers + side_number_additions
    solver.set_side_numbers(list(side_numbers))
    down_index = int(np.min(side_numbers))
    up_index = int(np.max(side_numbers))
    print(f"down_index {down_index} up_index {up_index}")
    return down_index, up_index

def run_f_star_lr(solver, experiment_name="", continue_from=0, visualize=False):
    print(f"Visualize {visualize}")
    # solver.solve_f_star(num_iterations=20000, spring_constant=10.0, o=0.0, i_round=-1, visualize=visualize)
    
    print(f"Running f_star solver lr")
    save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_f_star_0.bin"
    if continue_from == 0:
        # remove and remake dirs python_angles to save the solver visualization
        os.system("rm -rf python_angles")
        os.system("mkdir python_angles")
        for _ in range(4):
            solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=3.0, other_block_factor=0.5, lr=0.25, error_cutoff=-1.0, display=visualize)
            solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=2.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=visualize)
            solver.solve_f_star_with_labels(num_iterations=15000, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=visualize)
        solver.save_graph(save_path)
    elif continue_from == 1:
        solver.load_graph(save_path)

    save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_f_star_1.bin"
    if continue_from <= 1:
        solver.solve_f_star_with_labels(num_iterations=200000, spring_constant=1.0, other_block_factor=0.1, lr=0.05, error_cutoff=-1.0, display=visualize)
        solver.save_graph(save_path)
    else:
        solver.load_graph(save_path)

    # generate video
    if continue_from < 2:
        try:
            solver.generate_video("python_angles", f"experiments/{experiment_name}/python_angles_setup.avi", 10)
            compress_video(f"experiments/{experiment_name}/python_angles_setup.avi", f"experiments/{experiment_name}/python_angles_setup.mp4", codec='mp4v', scale=0.5)
        except:
            print(f"Error compressing video experiments/{experiment_name}/python_angles_setup.avi")

    return 11

def run_f_star(solver, experiment_name="", continue_from=0, visualize=False):
    print(f"Running f_star solver")
    print(f"Visualize {visualize}")
    save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_f_star_0.bin"
    if continue_from == 0:
        # remove and remake dirs python_angles to save the solver visualization
        os.system("rm -rf python_angles")
        os.system("mkdir python_angles")
        # solver.solve_f_star(num_iterations=100000, spring_constant=5.0, o=0.0, i_round=-1, visualize=visualize)
        solver.solve_f_star(num_iterations=75000, spring_constant=1.0, o=0.0, step_sigma=36000000.0, i_round=0, visualize=visualize)
        solver.save_graph(save_path)
    elif continue_from == 1:
        solver.load_graph(save_path)
    
    save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_f_star_1.bin"
    if continue_from <= 1:
        # solver.solve_f_star(num_iterations=10000, spring_constant=4.5, o=0.000, i_round=0, visualize=visualize)
        # solver.solve_f_star(num_iterations=10000, spring_constant=4.0, o=0.00, i_round=1, visualize=visualize)
        # solver.solve_f_star(num_iterations=20000, spring_constant=3.25, o=0.0, i_round=2, visualize=visualize)
        # solver.solve_f_star(num_iterations=20000, spring_constant=2.50, o=0.0, i_round=3, visualize=visualize)
        # solver.solve_f_star(num_iterations=20000, spring_constant=1.75, o=0.0, i_round=4, visualize=visualize)
        # solver.solve_f_star(num_iterations=20000, spring_constant=1.5, o=0.0, i_round=5, visualize=visualize)
        solver.solve_f_star(num_iterations=15000, spring_constant=1.0, step_sigma=360.0, o=0.0, i_round=1, visualize=visualize)
        # solver.solve_f_star(num_iterations=15000, spring_constant=0.75, o=0.02, i_round=7, visualize=visualize)
        # solver.solve_f_star(num_iterations=15000, spring_constant=0.8, o=0.02, i_round=8, visualize=visualize)
        # solver.solve_f_star(num_iterations=15000, spring_constant=0.9, o=0.02, i_round=9, visualize=visualize)
        # solver.solve_f_star(num_iterations=15000, spring_constant=1.1, o=0.0, i_round=10, visualize=visualize)
        solver.save_graph(save_path)
    else:
        solver.load_graph(save_path)

    # generate video
    if continue_from < 2:
        try:
            solver.generate_video("python_angles", f"experiments/{experiment_name}/python_angles_setup.avi", 10)
            compress_video(f"experiments/{experiment_name}/python_angles_setup.avi", f"experiments/{experiment_name}/python_angles_setup.mp4", codec='mp4v', scale=0.5)
        except:
            print(f"Error compressing video experiments/{experiment_name}/python_angles_setup.avi")

    return 2

def run_ring(solver, experiment_name="", i_round=11, fresh_start=0, standard_winding_direction=False, scale_left=1.0, scale_right=1.0):
    print(f"Running ring solver with fresh_start {fresh_start}")
    # remove and remake dirs python_sides to save the solver visualization
    if fresh_start == 0:
        os.system("rm -rf python_sides")
        os.system("mkdir python_sides")
    other_block_factor = 10.0
    solver.unfix() # unfix all nodes and get initial sides
    solver.reset_sides() # reset sides for a clean run
    solver.reset_happiness() # reset happiness for a clean run

    # solver parameters
    std_target = 0.013
    std_target2 = 0.205 # 0.075
    iter_target2 = 30000

    intermediate_ring_solution_save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_solver_connected_1.bin"
    if fresh_start <= 0:
        # warmup iterations
        solver.solve_ring(num_iterations=5 * 40000, i_round=i_round, other_block_factor = 1 * other_block_factor, increase_same_block_weight=False, 
                          std_target=std_target, standard_winding_direction=standard_winding_direction)
        solver.solution_loss()
        solver.save_graph(intermediate_ring_solution_save_path)
    elif fresh_start == 1:
        solver.load_graph(intermediate_ring_solution_save_path)
    i_round += 1
    
    intermediate_ring_solution_save_path2 = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_solver_connected_2.bin"
    if fresh_start <= 1:
        std_target_step = (std_target2 - std_target) / iter_target2 # target std from 0.013 -> 0.1 within 30'000 iterations
        solver.solve_ring(num_iterations=iter_target2, i_round=i_round, other_block_factor = 15 * other_block_factor, increase_same_block_weight=False, 
                          std_target=std_target, std_target_step=std_target_step, standard_winding_direction=standard_winding_direction)
        solver.solution_loss()
        i_round += 1
        # converge with std target 2
        solver.solve_ring(num_iterations=100000, i_round=i_round, other_block_factor = 15 *other_block_factor, increase_same_block_weight=False, 
                          std_target=std_target2, standard_winding_direction=standard_winding_direction)
        solver.solution_loss()
        i_round += 1
        solver.save_graph(intermediate_ring_solution_save_path2)
    elif fresh_start == 2:
        solver.load_graph(intermediate_ring_solution_save_path2)
        i_round += 2

    intermediate_ring_solution_save_path3 = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_solver_connected_3.bin"
    if fresh_start <= 2:
        # final parameters + speedup in convergence (makes solution a little less accurate, but convergence towards perfect solution is faster)
        # solver.solution_loss()
        for i in range(10):
            solver.solve_ring(num_iterations=10000, i_round=i_round, other_block_factor=150 * other_block_factor, increase_same_block_weight=True, 
                              convergence_speedup=True, std_target=std_target2, wiggle=i<7, standard_winding_direction=standard_winding_direction, 
                              scale_left=scale_left, scale_right=scale_right) # for sure c-thresh is lower than mean 0.002187, max 0.045 # 100k iterations, still a little short, maybe 500k needed
            solver.solution_loss()
            i_round += 1
        solver.save_graph(intermediate_ring_solution_save_path3)
    elif fresh_start == 3:
        solver.load_graph(intermediate_ring_solution_save_path3)
        i_round += 10

    intermediate_ring_solution_save_path4 = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_solver_connected_4.bin"
    if fresh_start <= 3:
        # let the sides converge with the final parameters. (makes solution more accurate)
        solver.solve_ring(num_iterations=100000, i_round=i_round, other_block_factor=15 * other_block_factor, increase_same_block_weight=True, 
                          convergence_speedup=False, std_target=std_target2, wiggle=False, standard_winding_direction=standard_winding_direction, 
                          scale_left=scale_left, scale_right=scale_right)
        solver.solution_loss()
        i_round += 1
        solver.save_graph(intermediate_ring_solution_save_path4)
    elif fresh_start == 4:
        solver.load_graph(intermediate_ring_solution_save_path4)
        i_round += 1
    
    if fresh_start < 4:
        # generate video
        try:
            solver.generate_video("python_sides", f"experiments/{experiment_name}/python_sides.avi", 10)
            compress_video(f"experiments/{experiment_name}/python_sides.avi", f"experiments/{experiment_name}/python_sides.mp4", codec='mp4v', scale=0.5)
        except:
            print(f"Error compressing video experiments/{experiment_name}/python_sides.avi")

    # filter the edges based on the ring solution
    solver.filter_graph_ring()
    print("filtered graph")

    return i_round

def run_winding_number(solver, experiment_name="", i_round=15):
    print(f"Running winding number solver")
    # remove and remake dirs python_winding_numbers to save the solver visualization
    os.system("rm -rf python_winding_numbers")
    os.system("mkdir python_winding_numbers")
    # find seed node in a manually picked "nice" region of the scroll
    seed_node = find_seed_node(solver)
    other_block_factor = 15.0
    # start fixing sides
    # side_fix_nr = 20000 # lower this for more accurate but slower solutions !!!BIG KNOB TO TURN!!! (not anymore)
    side_fix_nr = 200000 # prefiltered graph, almost not fixing needed anymore
    step_size = 120
    side_fix_iterations = int(2.0 * step_size * (solver.get_nr_nodes() // side_fix_nr + 1))
    print(f"side_fix_iterations {side_fix_iterations}")
    solver.solve_winding_number(num_iterations=side_fix_iterations, i_round=i_round, seed_node = seed_node, other_block_factor = other_block_factor, side_fix_nr = side_fix_nr)
    solver.solution_loss(use_sides=False)
    # generate video
    try:
        solver.generate_video("python_winding_numbers", f"experiments/{experiment_name}/python_winding_numbers.avi", 10)
        compress_video(f"experiments/{experiment_name}/python_winding_numbers.avi", f"experiments/{experiment_name}/python_winding_numbers.mp4", codec='mp4v', scale=0.5)
    except:
        print(f"Error compressing video experiments/{experiment_name}/python_winding_numbers.avi")

def main(graph_path="graph_scroll5_january_unrolling_full_v2.bin", experiment_name=None, standard_winding_direction=True, fresh_start_star=0, fresh_start_ring=0, z_min=None, z_max=None, visualize=False):
    # z_min = 3000
    # z_max = 4000
    if experiment_name is None:
        experiment_name = os.path.basename(graph_path)[:-4]
    if z_min is not None and z_max is not None:
        solver = graph_problem_gpu_py.Solver(graph_path, z_min=z_min, z_max=z_max)
    elif z_min is not None:
        solver = graph_problem_gpu_py.Solver(graph_path, z_min=z_min)
    elif z_max is not None:
        solver = graph_problem_gpu_py.Solver(graph_path, z_max=z_max)
    else:
        solver = graph_problem_gpu_py.Solver(graph_path)

    if not standard_winding_direction:
        solver.invert_graph()

    # make experiment folder if it does not exist
    os.makedirs("experiments", exist_ok=True)
    # make experiment folder if it does not exist
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    # make checkpoint folder in experiment folder
    os.makedirs(f"experiments/{experiment_name}/checkpoints", exist_ok=True)

    # fresh_start_star = 0 # continue solve computation from fresh_start
    i_round = run_f_star(solver, experiment_name, continue_from=fresh_start_star, visualize=visualize)
    # i_round = run_f_star_lr(solver, experiment_name, continue_from=fresh_start_star, visualize=visualize)
    solver.filter_f_star() # filter the graph based on f_star solution
    solver.generate_ply(f"experiments/{experiment_name}/solved_f_star.ply")

    # #  f* was on complete scroll, take subset for sides now to debug
    # solver.set_z_range(z_min, z_max)

    solver.largest_connected_component()
    save_path = f"experiments/{experiment_name}/checkpoints/checkpoint_graph_f_star_final.bin"
    solver.save_graph(save_path)

    if False: # not needed anymore with the manual Graph Labeling GUI
        # solver.solve_union()
        # exit()

        # fresh_start_ring = 0 # continue solve computation from fresh_start
        scale_outer = 1.0 # S5 right side
        scale_inner = 1.0 # S5 left side
        i_round = run_ring(solver, experiment_name, fresh_start=fresh_start_ring, i_round=i_round, standard_winding_direction=True, scale_left=scale_outer, scale_right=scale_inner)
        solver.solution_loss()

        if fresh_start_ring <= 4:
            run_winding_number(solver, experiment_name, i_round=i_round)
            solver.save_graph(f"experiments/{experiment_name}/checkpoints/checkpoint_graph_final.bin")

            #load graph
            solver.load_graph(f"experiments/{experiment_name}/checkpoints/checkpoint_graph_final.bin")
            solver.generate_ply(f"experiments/{experiment_name}/output_graph.ply")

            solution_path = f"experiments/{experiment_name}/output_graph.bin"
            solver.save_solution(solution_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Solve graph problem")
    parser.add_argument("graph_path", type=str, help="Path to the input graph .bin file")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment", default=None)
    parser.add_argument("--z_min", type=int, help="Minimum z value to consider", default=None)
    parser.add_argument("--z_max", type=int, help="Maximum z value to consider", default=None)
    parser.add_argument("--inversed_winding_direction", action="store_true", help="Use inverse winding direction. For example Scroll 1.")
    parser.add_argument("--fresh_start_star", type=int, default=0, help="Fresh start for f_star solver")
    parser.add_argument("--fresh_start_ring", type=int, default=0, help="Fresh start for ring solver")
    parser.add_argument("--visualize", action="store_true", help="Visualize Solver")

    args = parser.parse_args()
    print(f"Solver arguments: {args}")

    # Run the solver
    main(args.graph_path, experiment_name=args.experiment_name, standard_winding_direction=not args.inversed_winding_direction, fresh_start_star=args.fresh_start_star, fresh_start_ring=args.fresh_start_ring, z_min=args.z_min, z_max=args.z_max, visualize=args.visualize)
