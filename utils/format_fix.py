import os


def fix_trailing_whitespace(file_path):
    """Remove trailing whitespace from a file."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        for line in lines:
            file.write(line.rstrip() + "\n")


def process_files(directory):
    """Recursively fix Python files in the directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                fix_trailing_whitespace(file_path)


# Specify the directory to process
directory_to_fix = "../"  # Replace with your project's root directory
process_files(directory_to_fix)
print("Trailing whitespace removed!")
