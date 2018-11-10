from __future__ import print_function
from embeddings import FastTextEmbedding
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv





def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)
    #plt.show()

# Loading the vectors using fasttext in en,fr and de, default lang = 'en'

ftEnglish = FastTextEmbedding(default='zero')
#ftFrench = FastTextEmbedding(lang='fr', show_progress=True, default='zero')
ftGerman = FastTextEmbedding(lang='de', show_progress=True, default='zero')

deVocab = {}
enVocab = {}

#Open the dictionary file and get the Vocab for English and German
deenDictFile = "E:/Data/Translation/trans-dict/de-en.txt"
endeDictFile = 'E:/Data/Translation/trans-dict/en-de.txt'

#open file to write the embeddings from dict
#embedFile = open("de-en-EmbeddingsFromDict.csv","w", encoding='utf-8')
embedFile = open("de-en-EmbedSmall.csv","w", encoding='utf-8')

#process the dictionaries and get the appropriate word embeddings
#use zero encodings for out of vocabulary words.

with open(deenDictFile, 'r', encoding='utf-8') as csvf:
    csvreader = csv.reader(csvf,delimiter=' ')
    X = []
    Y = []
    i=0
    for row in csvreader:
        #get the german encoding and the equivalent english encoding and print to de-en encoding file
        deEmb = ftGerman.emb(row[0])
        enEmb = ftEnglish.emb(row[1])
        embedFile.write("%s," % row[0])
        embedFile.write(" ".join(str(x) for x in deEmb))
        embedFile.write(",%s," % row[1])
        embedFile.write(" ".join(str(x) for x in enEmb))
        embedFile.write("\n")
        if (i >=3200):
            break
embedFile.close()  





'''       
embeddingG = np.array(embeddingG)

#print(embedding)
# Creating the tsne plot [Warning: will take time]
tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

low_dim_embeddingE = tsne.fit_transform(embeddingE)
low_dim_embeddingF = tsne.fit_transform(embeddingF)
low_dim_embeddingG = tsne.fit_transform(embeddingG)
print(low_dim_embeddingE)
num_words = len(low_dim_embeddingE)
# Finally plotting and saving the fig 
plot_with_labels(low_dim_embeddingE, words, "English.png")
plot_with_labels(low_dim_embeddingF, words, "French.png")
plot_with_labels(low_dim_embeddingG, words, "German.png")

'''

