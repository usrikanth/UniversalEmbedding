import os


years = [1,2,3,4,5,6,7,8,9,10]

with open('year.txt', 'w') as file:
    for year in years:
        file.write("%i " % year)