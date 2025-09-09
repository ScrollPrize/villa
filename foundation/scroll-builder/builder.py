import docker
import os
import yaml
import re
from tqdm import tqdm
from copy import deepcopy
import argparse

class ScrollBuilder():
    """
    Building the derived scroll data formats from the raw data formats.
    """
    def __init__(self, config_file='builder.yaml'):
        self.client = docker.from_env()
        self.config_file = config_file
        self.config = self.load_yaml(self.config_file)
        self.base_path = self.config.get('base_path', os.getcwd())
        self.runtime = self.config.get('runtime', 'nvidia')
        self.track_path = "tracked_paths_versions.yaml"
        self.tracked_paths = self.load_yaml(self.track_path)

    def load_yaml(self, file):
        # Load the yaml file
        try:
            with open(file, 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            return {}
        
    def save_yaml(self, file, data):
        # Save the yaml file
        with open(file, 'w') as f:
            yaml.dump(data, f)
        
    def build_regex(self, variables, input_patterns={}):
        # parses the configuration arguments to valid regex patterns
        def replace_regex(regex, patterns):
            # Replace any placeholders with the completed regex patterns of earlier variables
            while any(f"${{{name}}}" in regex for name in patterns):
                for name, pattern in patterns.items():
                    regex = regex.replace("${" + name + "}", pattern)
            return regex
        
        completed_patterns = {}
        for key, regex in variables.items():
            # Replace any placeholders with the completed regex patterns of earlier variables
            regex = replace_regex(regex, input_patterns)
            regex = replace_regex(regex, completed_patterns)
            # Encapsulate in named groups to reference directly
            completed_patterns[key] = f"(?P<{key}>{regex})"
        return completed_patterns
    
    def fill_regex(self, regex, previous_match_result):
        """
        Fill a compiled regex pattern using previous regex match results.

        Parameters:
            regex: The regex pattern to use for matching.
            previous_match_result (re.Match): A regex match object containing previously captured groups.

        Returns:
            str: The filled regex pattern.
        """
        # Construct the regex dynamically based on previous matches
        filled_regex = regex
        for name, value in previous_match_result.groupdict().items():
            # Replace the named group in the regex with the captured value from previous_match_result
            # Use re.escape to ensure that any special characters in the value are escaped properly
            if value is not None:
                filled_regex = filled_regex.replace(f"(?P<{name}>.*?)", re.escape(value))
        return filled_regex
    
    def search_filesystem(self, regex_pattern):
        # Search the filesystem for files matching the regex pattern
        matches = set()
        compiled_regex = re.compile(regex_pattern)
        for dirpath, dirnames, filenames in tqdm(os.walk(self.base_path)):
            # remove base part of path
            dirpath = dirpath[len(self.base_path)+1:]
            for filename in sorted(filenames):
                full_path = os.path.join(dirpath, filename)
                match = compiled_regex.search(full_path)
                if match:
                    match_dict = match.groupdict()
                    # delete key "permutation" from groupdict
                    del match_dict["permutation"]
                    matches.add(tuple(match_dict.items()))
        matches = [{key: value for key, value in match} for match in matches]
        return matches
    
    def match_permutations(self, permutations, base_patterns):
        # Recursively match all permutations
        if len(permutations) == 0:
            return [base_patterns]
        permutation = permutations[0]
        permutation_patterns = self.build_regex({"permutation": permutation}, base_patterns)
        permutation_pattern = permutation_patterns["permutation"]
        # Match permutation and update base patterns
        matches = self.search_filesystem(permutation_pattern)
        # Recurse on remaining permutations
        matched_patterns = []
        for match in matches:
            base_patterns_ = deepcopy(base_patterns)
            base_patterns_.update(match)
            matched_patterns += self.match_permutations(permutations[1:], base_patterns_)

        # Find unique matched patterns dicts
        matched_patterns = [dict(s) for s in set(frozenset(d.items()) for d in matched_patterns)]
        return matched_patterns

    def script_recomputation(self, script_config):
        recompute_allways = script_config.get('recompute_allways', False)
        if recompute_allways:
            return True
        recompute_untracked = script_config.get('recompute_untracked', True)
        on_change_paths = script_config.get('on_change', [])
        for path in on_change_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist.")
            # Get last modified time
            last_modified = os.path.getmtime(path)
            # Check if path is tracked
            if path not in self.tracked_paths:
                if recompute_untracked:
                    return True
            # Check if the path was modified since last computation
            elif last_modified != self.tracked_paths[path]:
                return True
        return False
    
    def update_tracked_paths(self, script_config):
        on_change_paths = script_config.get('on_change', [])
        for path in on_change_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist.")
            # Get last modified time
            last_modified = os.path.getmtime(path)
            # Add to tracked paths
            self.tracked_paths[path] = last_modified
        # Save Updated Tracked Paths
        self.save_yaml(self.track_path, self.tracked_paths)
    
    def build_command(self, command_str, script_config):
        # Build the command, replace all variables with the matched patterns
        for key, value in script_config.items():
            try:
                command_str = command_str.replace("${" + key + "}", value)
            except:
                continue
        return command_str
    
    def build_commands(self, script, matched_patterns):
        # Builds docker and scrip commands
        script_configurations = []
        # All unique execution permutations
        for i, matched_pattern in enumerate(matched_patterns):
            script_configuration = {key: value for key, value in script.items()} # add all script configurations
            # Build the on_change paths
            on_change = deepcopy(script['on_change'])
            on_change_paths = []
            for path in on_change:
                on_change_paths.append(self.build_command(deepcopy(path), matched_pattern))
            script_configuration['on_change'] = on_change_paths
            # Build the commands
            commands = deepcopy(script['commands'])
            script_configuration['commands'] = []
            for command in commands:
                command_dict = {}
                # Workspace
                command_dict['workspace'] = command.get('workspace', '')
                # Docker command
                docker_command = command['docker_command']
                for volume in docker_command['volumes']:
                    for key, value in volume.items():
                        volume[key] = self.build_command(deepcopy(value), matched_pattern)
                command_dict['docker_command'] = docker_command
                # Script commands
                script_commands = command['script_commands']
                script_commands_list = []
                for script_command in script_commands:
                    script_commands_list.append(self.build_command(deepcopy(script_command), matched_pattern))
                command_dict['script_commands'] = script_commands_list
                script_configuration['commands'].append(command_dict)
            script_configurations.append(script_configuration) 
        # Unique script configurations
        return script_configurations
    
    def get_script_commands(self, script):
        # Finds all script path configurations
        base_patterns = self.build_regex(script['variables'])
        # Add base path to base patterns
        base_patterns['base_path'] = self.base_path
        # Returns all permutations of the script configurations for execution of the script
        matched_patterns = self.match_permutations(script['permutations'], base_patterns)
        print(f"Matched Patterns: {matched_patterns}")
        built_commands = self.build_commands(script, matched_patterns)
        return built_commands
    
    def run_docker_container(self, docker_command):
        # Environment variables
        docker_envs = docker_command['environment']
        environment = {
            key: os.getenv(value, value) for key, value in docker_envs.items()
        }

        # Volumes
        docker_volumes = docker_command['volumes']
        volumes = {
            volume_dict['host_path']: {"bind": volume_dict['container_path'], "mode": "rw" if volume_dict.get('write_access', False) else "ro"}for volume_dict in docker_volumes
        }

        # Run the Docker container
        try:
            print(f"Starting container {docker_command['name']}")
            container = self.client.containers.run(
                docker_command['name'],
                "tail -f /dev/null",  # Keeps the container running for executing further commands
                runtime=self.runtime,  # This is needed to utilize GPU optionally
                shm_size='150g',  # Setting the shared memory size to 150GB
                environment=environment,
                volumes=volumes,
                detach=True,
                remove=True,  # Equivalent to --rm
                auto_remove=True  # Automatically remove the container when it exits
            )
            print("Container started successfully.")
            return container
        except Exception as e:
            print(f"Failed to start the container: {e}")
            return None

    def execute_script_inside_container(self, container, script_commands, workspace):
        try:
            for command in script_commands:
                print(f"Executing command inside container: {command}")
                exec_log = container.exec_run(command, workdir=workspace)
                print("Output:", exec_log.output.decode())
        except Exception as e:
            print("Failed to execute script inside Docker:", e)

    def run_script(self, script_config):
        # Check if the script needs to be recomputed
        if not self.script_recomputation(script_config):
            return
        
        for command in script_config['commands']:
            # Extract docker command and script commands from the configuration
            docker_command = command['docker_command']
            script_commands = command['script_commands']
            workspace = command.get('workspace', '')

            # Start Docker container
            container = self.run_docker_container(docker_command)
            if container:
                # Execute scripts inside container
                self.execute_script_inside_container(container, script_commands, workspace)
    
    def build(self):
        # Build all the output data formats
        for script in self.config['scripts']:
            try:
                script_commands = self.get_script_commands(self.config['scripts'][script])
                for script_config in tqdm(script_commands, desc=f"Building {script}"):
                    try:
                        self.run_script(script_config)
                        # Track the paths
                        self.update_tracked_paths(script_config)
                    except:
                        pass
            except Exception as e:
                print(f"Failed to build {script} with error: {e}")
# Main
if __name__ == '__main__':
    # argparse config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='builder.yaml', help='Configuration file for the builder.')

    args = parser.parse_args()

    builder = ScrollBuilder(config_file=args.config)
    builder.build()

# Example: python3 builder.py
# Stop all containers: sudo docker rm -f $(sudo docker ps -aq)