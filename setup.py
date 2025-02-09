from setuptools import setup, find_packages
packages = find_packages(where=".")
#print(packages)
# attrs = {
#     'DEBUG': True,
# }
setup(
    name="RBS_network_models",
    version="0.0.3",
    packages=find_packages(where="."),  # Explicitly search from the root
    package_dir={"": "."},  # Map the root directory as the base
    include_package_data=True,  # Ensure all package data is included
    install_requires=[],  # Add any dependencies if needed
    #**attrs,
)