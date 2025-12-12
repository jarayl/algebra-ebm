#!/usr/bin/env python3

# Script to add get_equation_pair methods to dataset classes
import sys

def add_get_equation_pair_methods():
    """Add get_equation_pair methods to all dataset classes."""
    
    with open('src/algebra/algebra_dataset.py', 'r') as f:
        content = f.read()
    
    method_code = """
    def get_equation_pair(self, index: int) -> Tuple[str, str]:
        \"\"\"Get equation pair as strings for calibration and testing.\"\"\"
        if index >= len(self.equation_pairs):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.equation_pairs)}")
        return self.equation_pairs[index]
"""
    
    # Replace each get_problem_info with get_equation_pair + get_problem_info
    content = content.replace(
        "    def get_problem_info(self, index: int) -> Dict:",
        method_code + "\n    def get_problem_info(self, index: int) -> Dict:"
    )
    
    # Write back the modified file
    with open('src/algebra/algebra_dataset.py', 'w') as f:
        f.write(content)
    
    print("Successfully added get_equation_pair methods to all dataset classes")

if __name__ == "__main__":
    add_get_equation_pair_methods()