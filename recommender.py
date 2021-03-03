from processData import processData

inputFile = "data/ml-100k/u.data"

check = processData(inputFile).export_data()
for i in check:
    if i == "users":
        print(i)



