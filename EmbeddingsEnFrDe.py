from __future__ import print_function
from embeddings import FastTextEmbedding
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np




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
ftFrench = FastTextEmbedding(lang='fr', show_progress=True, default='zero')
ftGerman = FastTextEmbedding(lang='de', show_progress=True, default='zero')

vocab = ['canada', 'vancouver', 'toronto','Africa','Europe','Asia']
# Getting tokens and vectors
words = []
embeddingE = []
embeddingF = []
embeddingG = []
# Limit number of tokens to be visualized
limit = 500
vector_dim = 300


for word in vocab:
    embdEn = ftEnglish.emb(word)
    # Appending the vectors 
    embeddingE.append(embdEn)
    embdFr = ftFrench.emb(word)
    embeddingF.append(embdFr)
    embdDe = ftGerman.emb(word)
    embeddingG.append(embdDe)

    # Getting token 
    words.append(word)



# Reshaping the embedding vector 
embeddingE = np.array(embeddingE)
embeddingF = np.array(embeddingF)
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



