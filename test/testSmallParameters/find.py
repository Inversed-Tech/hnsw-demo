##### this code is used for reading the time and found number in 
##### the txt file of the test_with_small_parameters.txt
import re

def extract_numeric_values_from_content(content):
    """
    Extracts all numeric values (integers and floats) from the given content string.

    :param content: A string containing the content to extract numbers from.
    :return: A list of all numeric values found in the content.
    """
    # Use regular expressions to find all numeric values in the content
    numeric_values = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    
    # Convert all the found values to float if they contain a decimal point, otherwise to int
    numeric_values = [float(num) if '.' in num else int(num) for num in numeric_values]
    
    return numeric_values

def find_block_with_context(file_path, search_content, before=1, after=8):
    """
    Finds a block of lines containing specific content in a file and prints the line
    before and eight lines after the matched block.

    :param file_path: Path to the input text file.
    :param search_content: List of lines to search for as a block.
    :param before: Number of lines to print before the matched block (default is 1).
    :param after: Number of lines to print after the matched block (default is 8).
    """
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine the length of the search content block
    block_length = len(search_content)

    count = 0
    time = 0
    found = []
    timer = []   
    # Iterate through the file lines
    for i in range(len(lines) - block_length + 1):
        # Extract the current block from the file
        current_block = lines[i:i + block_length]
        # Compare the current block with the search content block
        if all(search_line.strip() in current_block[j].strip() for j, search_line in enumerate(search_content)):
            print(f"\nMatch found starting at line {i + 1}:")

            # Print the line before the matched block if it exists
            if i - before >= 0:
                # print("\n-- Line before the match --")
                # print(f"Line {i}: {lines[i - before].strip()}")
                # print(f"{lines[i - before].strip()}")
                found.append(lines[i - before].strip())
            if lines[i - before].strip() == 'Found':
                count += 1   

            # # Print the matched block
            # print("\n-- Matched block --")
            # for j in range(block_length):
            #     print(f"Line {i + j + 1}: {lines[i + j].strip()}")

            # Print the lines after the matched block
            # print("\n-- Lines after the match --")
            end_index = min(len(lines), i + block_length + after)
            # for j in range(i + block_length, end_index):
            #     print(f"Line {j + 1}: {lines[j].strip()}")
            # print(f"Line {i + block_length + 8}: {lines[end_index-1].strip()}")
            after_line = extract_numeric_values_from_content(lines[end_index-1].strip())
            # print(f"{after_line[0]}")
            timer.append(after_line[0])
            time += float(after_line[0])
    print("\n\nMatch: ")
    for i in found:
        if i == 'Found':
            print(1)
        else:
            print(0)    
    print("\n\nTime: ")
    for i in timer:
        print(i)
    
    print("\n\nMatch: " + f"{(count*100)/6}")
    print("Time: " + f"{time/6}")        

#### 
## when someone runs this code with test.txt, use the following parameters (7 loops)
## M = 8
## efConstruction = 8, 16
## efSearch = 8, 16, 32, 64, 128, 256, 512 (efs=64,128,256,512 5 experiments occured) 
## k = 005, 010, 020, 050, 100, 200, 500
file_path = 'test_with_small_parameters.txt'  # Replace with the path to your text file
search_content = [
    'M                : 8',
    'efConstruction   : 16',
    'efSearch         : 8',  
    'm_L              : 0.48',
    'K                : 005',   
]  # Replace with the content lines you want to search for

find_block_with_context(file_path, search_content, before=1, after=8)
