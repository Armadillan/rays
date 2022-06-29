import os
#from collections import Counter
import re

data = {}

regex = re.compile(r"0(\.|,)(0)*(\d)")

files = [f for f in os.listdir(".") if f[-4:] == ".csv" and f != "fd_output.csv"]

output = ","

for filename in files:
    with open(filename) as file:
        digits = {str(x): 0 for x in range(10)}
        for line in file.readlines():
            match = regex.match(line)
            if match:
                digits[match.group(3)] += 1
            else:
                digits[line[0]] += 1
        #digits = [line[0] for line in file.readlines()]
        #data[filename] = Counter(digits)
        data[filename] = digits

    output += filename + ","

output = output[:-1]
output += "\n"

totals = {}

output += "# values,"
for filename in files:
    total = sum(data[filename].values())
    output += str(total) + ","
    totals[filename] = total

output = output[:-1]
output += "\n"


for i in range(10):
    output += str(i) + ","
    for filename in files:
        output += str(data[filename][str(i)] / totals[filename]) + ","
    output = output[:-1]
    output += "\n"

output = output[:-1]

with open("fd_output.csv", "w") as file:
    file.write(output)
