import os
import sys
from pprint import pprint

#use git to get root directory of repository
def get_git_root():
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    return git_root

def get_git_root_ws():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Root directory check
        if os.path.isdir(os.path.join(current_dir, '.git')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def add_submodule_paths():
    workspace_folder = get_git_root_ws()
    submodule_folders = [f.path for f in os.scandir(workspace_folder) if f.is_dir() and f.name.startswith('submodules')]
    for submodule_folder in submodule_folders:
        for sub_submodule_folder in os.scandir(submodule_folder):
            if sub_submodule_folder.is_dir():
                sys.path.insert(0, sub_submodule_folder.path)
        #sys.path.insert(0, submodule_folder)

def set_pythonpath():
    pprint(sys.path)
    #workspace_folder = os.path.dirname(os.path.abspath(__file__))
    workspace_folder = get_git_root_ws()
    sys.path.insert(0, workspace_folder)
    add_submodule_paths()
    pprint(sys.path)
    #os.environ['PYTHONPATH'] = workspace_folder
    #sys.path.insert(0, workspace_folder)

if __name__ == "__main__":
    set_pythonpath()