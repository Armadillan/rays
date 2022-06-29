
data = []
with open("riksdagen2.csv") as file:
    lines = file.readlines()

for line in lines:
    if line != "\n":
        data.append(line)

with open("riksdagen.csv", "w") as file:
    for line in data:
        file.write(line)