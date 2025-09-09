# scroll-builder
Scroll-Builder facilitates server-side processing on scroll data, updating derived data whenever input data or computation scripts change. It utilizes Docker to ensure a consistent and isolated environment for computations.

Clone the repository:

```bash
git clone --recurse-submodules https://github.com/ScrollPrize/scroll-builder
```

## Configuration and Setup
The behavior of the Scroll-Builder is configured via a builder.yaml file, which dictates how scripts are executed based on the filesystem's state and the specifics of the data processing requirements.

## How to add a computation script

### Structure of builder.yaml
The builder.yaml file contains definitions for variables, data paths, and scripts that outline the computation processes. Here's an example of how this configuration might look:

```yaml
base_path: "/media/julian/2"
scripts:
  render_surface_volumes:
    variables:
      scroll: "[^/]*?.volpkg"
      path_id: "scroll-builder-test_[0-9]+"
      obj_id: "[0-9]+"
      path: "${scroll}/${path_id}"
    permutations:
      - "${path}/${obj_id}.obj"
    on_change:
      - "${base_path}/${path}/${obj_id}.obj"
      - "/home/julian/gitThaumato/ThaumatoAnakalyptor/ThaumatoAnakalyptor/mesh_to_surface.py"
    recompute_untracked: True
    recompute_allways: False
    commands:
      - command1:
        docker_command: 
          volumes: 
            -   host_path: "${base_path}/${scroll}"
                container_path: "/scroll.volpkg"
                write_access: False
            -   host_path: "${base_path}/${scroll}/${path_id}"
                container_path: "/scroll.volpkg/${path_id}"
                write_access: True
            -   host_path: "/home/julian/gitThaumato/ThaumatoAnakalyptor/"
                container_path: "/workspace"
                write_access: True
            -   host_path: "/tmp/.X11-unix"
                container_path: "/tmp/.X11-unix"
          environment:
            DISPLAY: "DISPLAY"
          name: "thaumato_image"
        script_commands: 
          - "python3 -m ThaumatoAnakalyptor.mesh_to_surface /scroll.volpkg/${path_id}/${obj_id}.obj /scroll.volpkg/volumes/scroll1_grids --display"
```

### Docker Configuration
Docker images are used to compute derived data. Ensure that your Docker setup includes a valid Dockerfile and that file access control is enforced by setting appropriate read-only or read-write permissions on mounted volumes.

### Adding a Computation Script
To add a new computation script:

- Define the script in the builder.yaml under the scripts section.
- Specify any necessary Docker settings such as image, volumes, and environment variables.
- List the arguments and execution flow influenced by regex-matched directories.

## Script Execution
Scripts are executed based on the configuration defined in builder.yaml. Each script runs if the input data or computation script changes, ensuring that all data transformations are up-to-date.

### Python Script (builder.py)
The Python script builder.py is responsible for loading configurations, building regex patterns, matching file system directories, and executing Docker containers with the specified environment. Here's an overview of its functionality:

- Configuration Loading: Reads and parses builder.yaml.
- Regex Building: Dynamically creates regex patterns to match filesystem paths.
- File System Searching: Searches for files matching the regex patterns.
- Docker Execution: Manages Docker containers to run specified scripts in an isolated environment.

### Running the Builder
To execute the builder, run the following command:
```bash
python3 builder.py
```

Stopping All Docker Containers
If you need to stop all running Docker containers and clean up after running scripts, use the following command:

```bash
sudo docker rm -f $(sudo docker ps -aq)

```

This command stops and removes all Docker containers, ensuring no residual containers are left running.