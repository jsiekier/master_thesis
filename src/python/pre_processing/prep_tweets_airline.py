import csv
inp= open("Tweets.csv","r",encoding="utf-8")
out= open("Out.csv","w",encoding="utf-8")
filereader = csv.reader(inp, delimiter=",",quotechar='\"')

for line in filereader:
    txt=line[0]+"\t"+line[1]+"\t"+line[10].replace("\t","").replace("\n","")+"\n"
    out.write(txt)
out.close()