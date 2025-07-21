# aw 2025-01-11 15:45:00 - notes to self. This is typical structure for developing a python package according to ChatGPT. Just something to keep in mind.

your_package/
├── your_package/             # Main package directory
│   ├── __init__.py           # Marks this directory as a package, can contain initialization code
│   ├── module1.py            # Your first module
│   ├── module2.py            # Your second module
│   └── subpackage/           # Subpackage (if needed)
│       ├── __init__.py
│       └── submodule.py
├── tests/                    # Tests directory
│   ├── __init__.py
│   ├── test_module1.py       # Unit tests for module1
│   ├── test_module2.py       # Unit tests for module2
├── docs/                     # Documentation (optional but recommended)
│   └── index.md
├── examples/                 # Examples (optional but useful for users)
│   └── example_usage.py
├── LICENSE                   # License file (e.g., MIT, Apache 2.0)
├── README.md                 # Readme file for project description
├── pyproject.toml            # Build system and dependency configuration (preferred)
├── setup.py                  # Setup script for packaging (optional if using pyproject.toml)
├── setup.cfg                 # Additional configuration (optional)
├── requirements.txt          # List of dependencies (optional)
├── MANIFEST.in               # Files to include in the package distribution
└── .gitignore                # Git ignore file

# aw 2025-02-09 15:17:36 - testing repo shift thing.
- ok, git push origin dev_branch works 
    - even though remote -v prints wrong links.

