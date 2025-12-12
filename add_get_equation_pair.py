#!/usr/bin/env python3

# Script to add get_equation_pair methods to dataset classes
import sys

def add_get_equation_pair_methods():
    """Add get_equation_pair methods to all dataset classes."""
    
    with open('src/algebra/algebra_dataset.py', 'r') as f:
        lines = f.readlines()
    
    method_code = [
        "\n",
        "    def get_equation_pair(self, index: int) -> Tuple[str, str]:\n",
        "        \"\"\"Get equation pair as strings for calibration and testing.\"\"\"\n",
        "        if index >= len(self.equation_pairs):\n",
        "            raise IndexError(f\"Index {index} out of range for dataset size {len(self.equation_pairs)}\")\n",
        "        return self.equation_pairs[index]\n"
    ]
    
    # Find lines where get_problem_info methods start
    target_lines = []
    for i, line in enumerate(lines):
        if "def get_problem_info(self, index: int) -> Dict:" in line:
            target_lines.append(i)
    
    print(f"Found get_problem_info methods at lines: {[i+1 for i in target_lines]}")
    
    # Insert get_equation_pair methods before each get_problem_info method
    offset = 0
    for line_idx in target_lines:
        insert_at = line_idx + offset
        for code_line in method_code:
            lines.insert(insert_at, code_line)
            offset += 1
    
    # Write back the modified file
    with open('src/algebra/algebra_dataset.py', 'w') as f:
        f.writelines(lines)
    
    print("Successfully added get_equation_pair methods to all dataset classes")

if __name__ == "__main__":
    add_get_equation_pair_methods()