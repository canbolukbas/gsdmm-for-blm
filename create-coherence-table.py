# construct coherence table as CSV given the output of main.py
import sys
import re
from openpyxl import Workbook

file_name = sys.argv[1]

with open(file_name, encoding="utf8") as f:
    output = f.read()

# results is a list of tuples of shape (coherence_value, K, alpha, beta)
results = re.findall(r'Coherence value is (.*?) for K=(.*?), alpha=(.*?) and beta=(.*).', output)

output_filename = "{}_gdsmm_coherence_values.xlsx".format(file_name)
workbook = Workbook()
sheet = workbook.active

sheet["A1"] = "Coherence Value"
sheet["B1"] = "K (# clusters)"
sheet["C1"] = "Alpha"
sheet["D1"] = "Beta"

for i in range(len(results)):
    sheet["A{}".format(i+2)] = results[i][0]
    sheet["B{}".format(i+2)] = results[i][1]
    sheet["C{}".format(i+2)] = results[i][2]
    sheet["D{}".format(i+2)] = results[i][3]

workbook.save(filename=output_filename)

