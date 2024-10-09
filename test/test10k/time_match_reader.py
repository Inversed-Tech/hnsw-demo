##### this code is used for reading the time and found number in 
##### the txt files of test10k

import csv
import re

# Step 1: Read data from specific lines in a text file
def read_specific_lines(file_path, line_numbers):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract only the specified lines
    selected_lines = [lines[i].strip() for i in line_numbers if i < len(lines)]
    return selected_lines

# Step 2: Write the processed data to a CSV file
def write_to_csv(data, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Step 2: Write the processed data to a CSV file with appending
def append_to_csv(data, csv_file_path):
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


# Step 3: Extract numeric values from the lines
def extract_numeric_values(lines):
    numeric_data = []
    for line in lines:
        # Find all numeric values in the line
        numbers = re.findall(r'\d+\.?\d*', line)
        # If there are numeric values, add them to the list
        if numbers:
            numeric_data.append(numbers)
    return numeric_data

# Replace with a test10k text file path that you want to read the time
text_file_path = 'test10k64m64c.txt'  
# Write an output CSV file path name
csv_file_path = 'time_reader.csv'  
 

line_numbers = []
for i in range(100): 
    line_numbers.append(6945 + 23*i) # linesTime-10k: 24, 2331, 4638, 6945, 9252
# print(line_numbers)

# Read specific lines from the text file
selected_lines = read_specific_lines(text_file_path, line_numbers)

data = extract_numeric_values(selected_lines)

# Write the processed data to a CSV file
write_to_csv(data, csv_file_path)


match_numbers = []
for i in range(5): 
    match_numbers.append(2307 + 2307*i) # linesMatch-10k: 2307
# print(match_numbers)

# Read specific lines from the text file
selected_match_lines = read_specific_lines(text_file_path, match_numbers)

data_match = extract_numeric_values(selected_match_lines)

# efSearch = 32, 64, 128, 256, 512, you can find them in the last 5 lines of the csv file.
# Append the processed data to a CSV file
append_to_csv(data_match, csv_file_path)
