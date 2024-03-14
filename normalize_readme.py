"""Used to normalize the readme file to capitalize the first word."""
import os
import re

with open('README.md', 'r') as f:
    data = f.readlines()
    
# Loop for each line and split space and only upper first word
res = []
for d in data:
    data_split = re.split(r'(\w+)', d)
    for i, w in enumerate(data_split):
        if w.isalpha():
            data_split[i] = w.capitalize()
            break
    res.append(''.join(data_split))
    
with open('README.md', 'w') as f:
    f.write(''.join(res))