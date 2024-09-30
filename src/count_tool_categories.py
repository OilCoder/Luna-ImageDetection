import os
from collections import Counter

def count_tool_categories():
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    classified_names_path = os.path.join(project_root, 'classified_tool_names.txt')

    category_counter = Counter()
    unknown_tools = []

    with open(classified_names_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2:
                filename, category = parts[0].strip(), parts[1].strip()
                category_counter[category] += 1
                if category == "Unknown":
                    unknown_tools.append(filename)

    print("Tool count by category:")
    for category, count in sorted(category_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count}")

    total_tools = sum(category_counter.values())
    print(f"\nTotal number of tools: {total_tools}")

    # print("\nTools classified as Unknown:")
    # for tool in unknown_tools:
    #     print(f"  {tool}")

if __name__ == "__main__":
    count_tool_categories()