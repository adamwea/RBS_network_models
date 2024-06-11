import subprocess
import re
import os

# Function to compile the SHMEM program
def compile_shmem_program():
    result = subprocess.run(
        ['mpicc', '-o', 'shmem_test', 'oshmem_symmetric_data.c', '-loshmem'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("Error compiling shmem program:", result.stderr)
        return False
    return True

def run_shmem_test(num_pes):
    # Run the SHMEM program using shmemrun
    result = subprocess.run(
        ['oshrun', '-n', str(num_pes), './shmem_test'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("Error running shmem program:", result.stderr)
        return False

    output = result.stdout
    print("Program Output:\n", output)
    return verify_output(output, num_pes, SIZE)

def verify_output(output, num_pes, size):
    expected_pattern = r"Target on PE (\d+) is\s+((?:\d+\s+)+)"
    matches = re.findall(expected_pattern, output)

    if len(matches) != (num_pes - 1):
        print(f"Expected {num_pes - 1} matches, but found {len(matches)}")
        return False

    for match in matches:
        pe, data = int(match[0]), match[1].strip()
        data_values = list(map(int, data.split()))
        expected_values = list(range(size))

        if data_values != expected_values:
            print(f"Data mismatch on PE {pe}: expected {expected_values}, got {data_values}")
            return False

    return True

if __name__ == "__main__":
    SIZE = 16
    NUM_PES = 4  # Number of processing elements

    # Compile the SHMEM program
    if not compile_shmem_program():
        print("Compilation failed. Exiting.")
        exit(1)

    # Run the SHMEM test
    if run_shmem_test(NUM_PES):
        print("SHMEM test passed successfully.")
    else:
        print("SHMEM test failed.")
