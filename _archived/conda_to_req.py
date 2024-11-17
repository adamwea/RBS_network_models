# import yaml

# # Load the YAML file
# with open('environment.yml', 'r') as f:
#     data = yaml.safe_load(f)

# # Extract package names and versions
# requirements = []
# for dep in data['dependencies']:
#     if isinstance(dep, str):
#         # if dep.startswith('python='):
#         #     continue  # Optionally skip python dependency
#         requirements.append(dep.replace('=', '=='))

# # Write the requirements.txt file
# with open('requirements.txt', 'w') as f:
#     for item in requirements:
#         f.write("%s\n" % item)

import yaml

# Load the YAML file
with open('environment.yml', 'r') as f:
    data = yaml.safe_load(f)

# Extract pip-installed packages from the dependencies
pip_packages = []
dependencies = data['dependencies']
for dep in dependencies:
    if isinstance(dep, dict) and 'pip' in dep:
        pip_packages = dep['pip']
        break

# Write the requirements.txt file with pip packages
with open('requirements.txt', 'w') as f:
    for package in pip_packages:
        f.write(f"{package}\n")