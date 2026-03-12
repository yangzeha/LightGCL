import json

notebook_path = 'lightgcl.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            
            # Handle list vs string source
            if isinstance(source, str):
                lines = source.splitlines(keepends=True)
                source_is_list = False
            else:
                lines = source
                source_is_list = True

            new_lines = []
            skip_mode = False
            
            for line in lines:
                stripped = line.strip()
                
                # Check for initialization lines (exact match or simple startswith)
                if stripped.startswith('prev_recall_20 = 0') or \
                   stripped.startswith('prev_ndcg_20 = 0') or \
                   stripped.startswith('prev_recall_40 = 0') or \
                   stripped.startswith('prev_ndcg_40 = 0'):
                    modified = True
                    continue # REMOVE line
                
                # START SKIPPING BLOCK
                if '# EARLY STOPPING CHECK' in line:
                    skip_mode = True
                    modified = True
                    continue # Remove marker line
                
                if skip_mode:
                    # Check if we should stop skipping
                    # Condition: Not empty line AND starts with non-whitespace
                    if stripped and not line[0].isspace():
                        skip_mode = False
                        new_lines.append(line)
                    else:
                        # Continue skipping (it's inside the block or empty line inside block)
                        pass
                else:
                    new_lines.append(line)
            
            if modified:
                if source_is_list:
                    cell['source'] = new_lines
                else:
                    cell['source'] = "".join(new_lines)

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("No changes found.")

except Exception as e:
    print(f"Error: {e}")
