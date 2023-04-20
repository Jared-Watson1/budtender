import json

dataFile = 1
dataFile = "data/character_data" + str(dataFile) + ".jsonl"

def addToDataFile(prompt, completion, file=dataFile):
    with open(file, "a") as f:
        data = {"prompt": "START\n" + prompt + "\nEND", "completion": "START\n" + completion + "\nEND"}
        f.write(json.dumps(data) + "\n")

def txtToDataFile(prompt, file="data/data.txt"):
    with open(file, 'r') as f:
        data = f.read()
    addToDataFile(prompt, data)

p = """
"""
txtToDataFile(p)



