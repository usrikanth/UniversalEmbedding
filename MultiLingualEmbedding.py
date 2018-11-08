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

# Loading the vectors using fasttext in en,fr and de

ftEnglish = FastTextEmbedding(default='zero')
#ftFrench = FastTextEmbedding(lang='fr', show_progress=True, default='zero')
ftGerman = FastTextEmbedding(lang='de', show_progress=True, default='zero')

deVocab = {}
enVocab = {}

#Open the dictionary file and get the Vocab for English and German
deenDictFile = "E:/Data/Translation/trans-dict/de-en.txt"
endeDictFile = 'E:/Data/Translation/trans-dict/en-de.txt'

#open file to write the embeddings from dict
#embedFile = open("de-en-EmbeddingsFromDict.csv","w")

#process the dictionaries and get the appropriate word embeddings
#use zero encodings for out of vocabulary words. Store them in a dict to look up already encoded words for word miss

with open(deenDictFile, 'r', encoding='latin-1') as csvf:
    csvreader = csv.reader(csvf,delimiter=' ')
    X = []
    Y = []
    for row in csvreader:
        #get the german encoding and the equivalent english encoding and print to de-en encoding file
        deEmb = ftGerman.emb(row[0])
        X.append(deEmb)
        enEmb = ftEnglish.emb(row[1])
        Y.append(enEmb)
        embedFile.write("%s, " % row[0])
        for deNum in deEmb:
            embedFile.write("%f "% deNum)
        embedFile.write(', %s,\t' % row[1])
        for enNum in enEmb:
            embedFile.write("%f " % enNum)
        embedFile.write("\n")
        





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
