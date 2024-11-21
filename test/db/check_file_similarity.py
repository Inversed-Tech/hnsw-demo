import filecmp
 
f1 = "check_file_similarity1.csv"
f2 = "check_file_similarity2.csv"
f3 = "check_file_similarity3.csv"
 
# shallow comparison
result = filecmp.cmp(f1, f2)
print(result)
# deep comparison
result = filecmp.cmp(f1, f2, shallow=False)
print(result)

# shallow comparison
result = filecmp.cmp(f1, f3)
print(result)
# deep comparison
result = filecmp.cmp(f1, f3, shallow=False)
print(result)

# shallow comparison
result = filecmp.cmp(f2, f3)
print(result)
# deep comparison
result = filecmp.cmp(f2, f3, shallow=False)
print(result)