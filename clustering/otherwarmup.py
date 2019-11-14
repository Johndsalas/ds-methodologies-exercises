import json

with open('./bayes.json') as f:
    bayes = json.load(f)

#print(bayes)


for key,value in bayes.items():

    if key == "Location":

        print("The location is: " + value)

     if key == "Floor":

        print("The Floor number is: " + str(value))