from word2vec_twitter_model import word2vecReader as w2vec# extract code from https://radimrehurek.com/gensim/models/word2vec.html
import os

input_file=open("data/filtered_se.csv","r")
output_file=open("data/word_vec_side_effects.txt","w",encoding="utf-8")

print("Load embeddings...")
w2v = w2vec.Word2Vec.load_word2vec_format(os.path.join("../data","path_to_model"), binary=True)
print("Embeddings loaded!")
se_set=set([])

se_counter=0

for side_effect in input_file:
    side_effect = side_effect.lower().replace("\n", "")
    if side_effect!="" and len(side_effect)>4:
        se_counter+=1
        se_set.add(side_effect)

    se=side_effect.replace(" ","_")
    if se in w2v.vocab:
        most_sim= w2v.most_similar(positive=[se],topn=25)
        for m in most_sim:
            if m[1]>0.7 and len(m[0])>4:
                #print(m[0],m[1],side_effect)
                se_set.add(m[0].replace("_"," "))

print(se_set)
print(len(se_set))
print(se_counter)
input_file.close()
for side_effect in se_set:
    output_file.write(side_effect+"\n")
output_file.close()
