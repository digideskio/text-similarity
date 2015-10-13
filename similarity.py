import os
import sys
import numpy as np
import itertools
from gensim import models
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics


def preprocess(sentences):
    """
    Some basic text preprocessing, removing line breaks, handling
    punctuation etc.
    """
    punctuation = """.,?!:;(){}[]"""
    sentences = [sent.lower().replace('\n','') for sent in sentences]
    sentences = [sent.replace('<br />', ' ') for sent in sentences]

    #treat punctuation as individual words
    for c in punctuation:
        sentences = [sent.replace(c, ' %s '%c) for sent in sentences]
    sentences = [sent.split() for sent in sentences]
    return sentences


def get_data(inputfile):
    """
    Fetching files and separating samples into sentences to compare
    """
    fname = inputfile.split('/')[-1]
    with open(inputfile,'r') as f:
        samples = f.readlines()
    sentences1 = []
    sentences2 = []
    for sample in samples:
        sample = sample.rstrip('\n')
        sample = sample.split('\t')
        sentences1.append(sample[0])
        sentences2.append(sample[1])
    return fname, samples, sentences1, sentences2


def labelize_text(sentences, label_type):
    """
    A special requirement for gensim doc2vec,
    each sentence has to be labeled and turned
    into LabeledSentence object.
    """
    labelized = []
    LabeledSentence = models.doc2vec.LabeledSentence

    for ind,sent in enumerate(sentences):
        label = '%s_%s'%(label_type,ind)
        labelized.append(LabeledSentence(sent, [label]))
    return labelized


def PCA_model(samples):
    """
    Alternative to Doc2Vec for data vectorization
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    svd = TruncatedSVD(n_components=5, random_state=42)
    pca = make_pipeline(vectorizer, svd, Normalizer(copy=False))
    model = pca.fit(samples)
    return model


def D2V_model(sentences1, sentences2):
    """
    Initializing and training a Doc2Vec model
    """
    corpus = sentences1+sentences2
    d2v = models.Doc2Vec(min_count=1, window=10, size=10, sample=1e-3, 
                                                        negative=5, workers=1)
    d2v.build_vocab(corpus)
    d2v.train(corpus)
    return d2v


def get_vecs(model, sentences, size):
    """
    Vectorizes input sentences using pre-trained Doc2Vec model
    and returns as numpy arrays.
    """
    vecs = [np.array(model[sent.labels[0]]).reshape((1, size)) for sent in sentences]
    return np.concatenate(vecs)


def main(argv):

    fname, samples, sentences1, sentences2 = get_data(argv[1]) 

    samples = preprocess(samples)
    sentences1 = preprocess(sentences1)
    sentences2 = preprocess(sentences2)

    print(' --- data fetching done ---')

    sentences1 = labelize_text(sentences1, 'TEXTONE')
    print('--- labelizing sentences1 done ---')
    sentences2 = labelize_text(sentences2, 'TEXTTWO')
    print('--- labelizing sentences2 done ---')

    model = D2V_model(sentences1, sentences2)
    print('--- Doc2Vec model training done ---')

    vecs1 = get_vecs(model, sentences1, 10)
    vecs2 = get_vecs(model, sentences2, 10)

    sims = []
    for vec1,vec2 in zip(vecs1,vecs2):
        sim = 1 - spatial.distance.cosine(vec1,vec2)
        sims.append(sim)

    with open('output/train/'+fname,'w') as fout:    
        for s in sims:
            scaled_sim= (5-0)/(max(sims)-min(sims))*(s-max(sims))+5
            print(scaled_sim, file=fout)

    print()
    print('--- File %s is done! ---', fname)



if __name__ == '__main__':
    main(sys.argv)