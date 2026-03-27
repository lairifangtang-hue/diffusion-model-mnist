
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import json

# Read the original notebook content
with open('/home/ubuntu/upload/pasted_file_eYviNH_A0123456J_Manus_Assignment_4.ipynb', 'r', encoding='utf-8') as f:
    original_notebook = nbformat.read(f, as_version=4)

# Read the GPU-enabled code from diffusion_model.py
with open('/home/ubuntu/diffusion_model_project/diffusion_model.py', 'r', encoding='utf-8') as f:
    gpu_code_content = f.read()

# Find the cell to replace and replace its content
# The target cell is the one with the 'Your code starts here' and 'Your code ends here' comments
found_and_replaced = False
for cell in original_notebook.cells:
    if cell.cell_type == 'code' and '# Your code starts here' in cell.source:
        # Replace the content of this cell with the GPU-enabled code
        cell.source = gpu_code_content
        found_and_replaced = True
        break

if not found_and_replaced:
    print("Error: Could not find the target code block in the original notebook.")
else:
    # Save the modified notebook
    with open('/home/ubuntu/A0123456J_Assignment_4_GPU.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(original_notebook, f)

    print("GPU-enabled Jupyter notebook generated successfully: A0123456J_Assignment_4_GPU.ipynb")
