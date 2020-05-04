from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfFeatureGenerator(FeatureGenerator):
    
    
    def __init__(self, name='tfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)

    def cosine_sim(self, x, y):
        try:
            if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
            if type(y) is np.ndarray: y = y.reshape(1, -1)
            d = cosine_similarity(x, y)
            d = d[0][0]
        except:
            print(x)
            print(y)
            d = 0.
        return d
    
    def process(self, df, header='train'):

        # 1). create strings based on ' '.join(Headline_unigram + articleBody_unigram) [ already stemmed ]
        def cat_text(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res
        df["all_text"] = list(df.apply(cat_text, axis=1))

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xHeadlineTfidf = vecH.fit_transform(df['Headline_unigram'].map(lambda x: ' '.join(x))) # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
        
        outfilename_htfidf = "feature_pkl/"+header+".headline.tfidf.pkl"
        with open(outfilename_htfidf, "wb") as outfile:
            pickle.dump(xHeadlineTfidf, outfile, -1)
        print('headline tfidf features of data set saved in %s' % outfilename_htfidf)
        
        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xBodyTfidf = vecB.fit_transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        print('xBodyTfidf.shape:', xBodyTfidf.shape)
        
        outfilename_btfidf = "feature_pkl/"+header+".body.tfidf.pkl"
        with open(outfilename_btfidf, "wb") as outfile:
            pickle.dump(xBodyTfidf, outfile, -1)
        print('body tfidf features of data set saved in %s' % outfilename_btfidf)
        
        # 4). compute cosine similarity between headline tfidf features and body tfidf features   
        simTfidf = np.asarray(list(map(self.cosine_sim, xHeadlineTfidf, xBodyTfidf)))[:, np.newaxis]
        print('simTfidf.shape:', simTfidf.shape)
        # simTfidfTrain = simTfidf[:n_train]
        outfilename_simtfidf = "feature_pkl/"+header+".sim.tfidf.pkl"
        with open(outfilename_simtfidf, "wb") as outfile:
            pickle.dump(simTfidf, outfile, -1)
        print('tfidf sim. features of data set saved in %s' % outfilename_simtfidf)
        
        return xHeadlineTfidf, xBodyTfidf, simTfidf

    def read(self, header='train'):

        filename_htfidf = "feature_pkl/%s.headline.tfidf.pkl" % header
        with open(filename_htfidf, "rb") as infile:
            xHeadlineTfidf = pickle.load(infile)

        filename_btfidf = "feature_pkl/%s.body.tfidf.pkl" % header
        with open(filename_btfidf, "rb") as infile:
            xBodyTfidf = pickle.load(infile)

        filename_simtfidf = "feature_pkl/%s.sim.tfidf.pkl" % header
        with open(filename_simtfidf, "rb") as infile:
            simTfidf = pickle.load(infile)

        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
        print('xBodyTfidf.shape:', xBodyTfidf.shape)
        print('simTfidf.shape:', simTfidf.shape)

        return [xHeadlineTfidf, xBodyTfidf, simTfidf.reshape(-1, 1)]

        # return [simTfidf.reshape(-1, 1)]

 #   Copyright 2017 Cisco Systems, Inc.
 #  
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #  
 #     http://www.apache.org/licenses/LICENSE-2.0
 #  
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
