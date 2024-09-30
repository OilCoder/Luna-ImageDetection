### Code to generate clasified labels using ChatGPT4, run on internal GTP machine, output: classified_tool_names_complete.txt. Labels may be changed
# # Let's load the new file uploaded by the user, classify the tools based on the categories, and then save the results.

# Path to the uploaded file
file_path_new = "file_names.txt"

# Reading the original file content
with open(file_path_new, "r") as file:
    original_lines_new = file.readlines()

general_categories = {
    "Pulling Tools": ["pulling", "spear", "GR type", "GS type", "SB type", "gr tool", "catch tool"],
    "Joints and Connectors": ["knuckle joint", "swivel joint", "quick joint", "connector", "sub connector", "joint", "coupling", "connection"],
    "Pressure Control Systems and Equipment": ["pressure control", "wellhead", "BOP", "blowout preventer", "grease injection", "stuffing box", 
                                               "adapter", "flange", "sealing", "union", "control head", "hydraulic_bop", "pressure_head"],
    "Centralizers": ["centralizer", "fluted centralizer", "roller centralizer", "guiding", "guiding head"],
    "Jars (Hydraulic or Mechanical)": ["jar", "mechanical jar", "hydraulic jar", "spring jar", "spang jar", "accelerator"],
    "Bailers": ["bailer", "sample bailer", "sand pump bailer", "hydrostatic bailer"],
    "Fishing Tools": ["fishing", "overshot", "spear", "retrieval", "fishing neck", "releasable", "wireline fishing tool"],
    "Gauge Cutters": ["cutter", "gauge cutter", "paraffin cutter", "scratcher"],
    "Spoolers": ["spooler", "spooling", "cable spooler", "winch", "spooling unit"],
    "Stem Tools": ["stem bar", "weight stem", "tungsten stem", "lead stem", "lead impression block", "tungsten_filled_stem", "lead_filled_stem"],
    "Running and Shifting Tools": ["running tool", "mandrel", "plug", "packer", "GS type", "SB type", "test tool", "shifting tool", "positioning_tool"],
    "Magnet Tools": ["magnet", "skirted magnet", "catcher", "retrieval magnet"],
    "Perforators and Wellbore Intervention Tools": ["perforator", "tubing perforator", "wellbore intervention"],
    "Hydraulic Tool Traps": ["hydraulic_tool_trap", "manual_locking", "hydraulic_control"],
    "Gauge Hangers and Pressure Gauges": ["gauge_hanger", "pressure_gauge", "unlatching_tool"],
    "Crossover Tools": ["crossover", "cross_over", "sub"],
    "Adapters and Extensions": ["adapter", "extension"],
    "Hay Pulleys": ["hay pulley", "hay_pulley"],
    "Torque and Torsion Tools": ["torsion tester", "torque tester"],
    "Test Caps": ["test cap"],
    "Stem Weights": ["stem weight", "weight bar"],
}

def classify_tool(tool_name):
    # Convert the tool name to lowercase for comparison
    tool_name_lower = tool_name.lower()
    # Check each category to see if any keyword matches the tool name
    for category, keywords in general_categories.items():
        for keyword in keywords:
            if keyword in tool_name_lower:
                return category
    # If no keyword matches, return "Unknown"
    return "Unknown"

# Create a list to store the classified content
classified_content_new = []

# Classify each tool based on the filename
for line in original_lines_new:
    filename = line.strip()  # Remove any surrounding whitespace or newlines
    category = classify_tool(filename)  # Classify the tool
    classified_content_new.append(f"{filename}: {category}\n")

# Write the classified content to a new file
output_file_path_new = "classified_tool_names.txt"
with open(output_file_path_new, "w") as output_file:
    output_file.writelines(classified_content_new)

output_file_path_new

import os

def read_file_names(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)

def read_classified_names(file_path):
    with open(file_path, 'r') as file:
        return set(line.split(':')[0].strip() for line in file)

def verify_file_names():
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)

    file_names_path = os.path.join(project_root, 'file_names.txt')
    classified_names_path = os.path.join(project_root, 'classified_tool_names.txt')

    file_names = read_file_names(file_names_path)
    classified_names = read_classified_names(classified_names_path)

    # Check for files in file_names.txt but not in classified_tool_names.txt
    missing_in_classified = file_names - classified_names
    if missing_in_classified:
        print("Files in file_names.txt but not in classified_tool_names.txt:")
        for name in sorted(missing_in_classified):
            print(f"  {name}")
    else:
        print("All files from file_names.txt are present in classified_tool_names.txt")

    # Check for files in classified_tool_names.txt but not in file_names.txt
    extra_in_classified = classified_names - file_names
    if extra_in_classified:
        print("\nFiles in classified_tool_names.txt but not in file_names.txt:")
        for name in sorted(extra_in_classified):
            print(f"  {name}")
    else:
        print("All files from classified_tool_names.txt are present in file_names.txt")

    if not missing_in_classified and not extra_in_classified:
        print("The file lists match perfectly!")

if __name__ == "__main__":
    verify_file_names()

