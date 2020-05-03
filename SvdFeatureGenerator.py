from FeatureGenerator import *
from TfidfFeatureGenerator import *
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class SvdFeatureGenerator(FeatureGenerator):


    def __init__(self, name='svdFeatureGenerator'):
        super(SvdFeatureGenerator, self).__init__(name)

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
        
        # n_train = df[~df['target'].isnull()].shape[0]
        # print 'SvdFeatureGenerator, n_train:',n_train
        # n_test  = df[df['target'].isnull()].shape[0]
        # print 'SvdFeatureGenerator, n_test:',n_test

        tfidfGenerator = TfidfFeatureGenerator('tfidf')
        featuresTrain = tfidfGenerator.read('train')
        xHeadlineTfidfTrain, xBodyTfidfTrain = featuresTrain[0], featuresTrain[1]
        
        xHeadlineTfidf = xHeadlineTfidfTrain
        xBodyTfidf = xBodyTfidfTrain
        # if n_test > 0:
        #     # test set is available
        #     featuresTest  = tfidfGenerator.read('test')
        #     xHeadlineTfidfTest,  xBodyTfidfTest  = featuresTest[0],  featuresTest[1]
        #     xHeadlineTfidf = vstack([xHeadlineTfidfTrain, xHeadlineTfidfTest])
        #     xBodyTfidf = vstack([xBodyTfidfTrain, xBodyTfidfTest])
	    
        # compute the cosine similarity between truncated-svd features
        svd = TruncatedSVD(n_components=50, n_iter=15)
        xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
        svd.fit(xHBTfidf) # fit to the combined train-test set (or the full training set for cv process)
        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
        xHeadlineSvd = svd.transform(xHeadlineTfidf)
        print('xHeadlineSvd.shape:', xHeadlineSvd.shape)
        
        # xHeadlineSvdTrain = xHeadlineSvd[:n_train, :]
        outfilename_hsvd = "feature_pkl/"+header+".headline.svd.pkl"
        with open(outfilename_hsvd, "wb") as outfile:
            pickle.dump(xHeadlineSvd, outfile, -1)
        print('headline svd features of data set saved in %s' % outfilename_hsvd)
        
        # if n_test > 0:
        #     # test set is available
        #     xHeadlineSvdTest = xHeadlineSvd[n_train:, :]
        #     outfilename_hsvd_test = "test.headline.svd.pkl"
        #     with open(outfilename_hsvd_test, "wb") as outfile:
        #         cPickle.dump(xHeadlineSvdTest, outfile, -1)
        #     print 'headline svd features of test set saved in %s' % outfilename_hsvd_test

        xBodySvd = svd.transform(xBodyTfidf)
        print('xBodySvd.shape:', xBodySvd.shape)
        
        # xBodySvdTrain = xBodySvd[:n_train, :]
        outfilename_bsvd = "feature_pkl/"+header+".body.svd.pkl"
        with open(outfilename_bsvd, "wb") as outfile:
            pickle.dump(xBodySvd, outfile, -1)
        print('body svd features of training set saved in %s' % outfilename_bsvd)
        
        # if n_test > 0:
        #     # test set is available
        #     xBodySvdTest = xBodySvd[n_train:, :]
        #     outfilename_bsvd_test = "test.body.svd.pkl"
        #     with open(outfilename_bsvd_test, "wb") as outfile:
        #         cPickle.dump(xBodySvdTest, outfile, -1)
        #     print 'body svd features of test set saved in %s' % outfilename_bsvd_test

        simSvd = np.asarray(list(map(self.cosine_sim, xHeadlineSvd, xBodySvd)))[:, np.newaxis]
        print('simSvd.shape:', simSvd.shape)

        # simSvdTrain = simSvd[:n_train]
        outfilename_simsvd = "feature_pkl/"+header+".sim.svd.pkl"
        with open(outfilename_simsvd, "wb") as outfile:
            pickle.dump(simSvd, outfile, -1)
        print('svd sim. features of data set saved in %s' % outfilename_simsvd)
        
        # if n_test > 0:
        #     # test set is available
        #     simSvdTest = simSvd[n_train:]
        #     outfilename_simsvd_test = "test.sim.svd.pkl"
        #     with open(outfilename_simsvd_test, "wb") as outfile:
        #         cPickle.dump(simSvdTest, outfile, -1)
        #     print 'svd sim. features of test set saved in %s' % outfilename_simsvd_test

        return xHeadlineSvd, xBodySvd, simSvd


    def read(self, header='train'):

        filename_hsvd = "feature_pkl/%s.headline.svd.pkl" % header
        with open(filename_hsvd, "rb") as infile:
            xHeadlineSvd = pickle.load(infile)

        filename_bsvd = "feature_pkl/%s.body.svd.pkl" % header
        with open(filename_bsvd, "rb") as infile:
            xBodySvd = pickle.load(infile)

        filename_simsvd = "feature_pkl/%s.sim.svd.pkl" % header
        with open(filename_simsvd, "rb") as infile:
            simSvd = pickle.load(infile)

        print('xHeadlineSvd.shape:', xHeadlineSvd.shape)
        print('xBodySvd.shape:', xBodySvd.shape)
        print('simSvd.shape:', simSvd.shape)

        return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]

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
