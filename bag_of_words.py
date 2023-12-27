import collections
import scipy.sparse as sp
import numpy as np

def bag_of_words():

  def tfidf(docs):
      """
      Create TFIDF matrix.  This function creates a TFIDF matrix from the
      docs input.

      Args:
          docs: list of strings, where each string represents a space-separated
                document

      Returns: tuple: (tfidf, all_words)
          tfidf: sparse matrix (in any scipy sparse format) of size (# docs) x
                 (# total unique words), where i,j entry is TFIDF score for 
                 document i and term j
          all_words: list of strings, where the ith element indicates the word
                     that corresponds to the ith column in the TFIDF matrix
      """


      all_words = {word for doc in docs for word in doc.split(" ")}
      wordIndexDict = {word:i for i,word in enumerate(list(all_words))}

      docWordCounts = {}
      for i,doc in enumerate(docs):
          words = doc.split(" ")
          counts = collections.Counter(words)
          docWordCounts[i] = dict(counts)
      df = {}

      for doc, wordCountDict in docWordCounts.items():
          for word in wordCountDict:
              if word in df:
                  df[word] += 1
              else:
                  df[word] = 1

      col  = np.array([wordIndexDict[word] for doc, wordCountDict in docWordCounts.items() for word in wordCountDict])
      row  = np.array([doc for doc, wordCountDict in docWordCounts.items() for word in wordCountDict])

      data = np.array([count*np.log(len(docs)*1.0/df[word]) for doc, wordCountDict in docWordCounts.items() for word,count in wordCountDict.items()])

      A = sp.coo_matrix((data, (row,col)))
      A = A.tocsr()
      A.eliminate_zeros()

      all_words = list(all_words)
      return A , all_words


      def cosine_similarity(X):
      """
      Return a matrix of cosine similarities.

      Args:
          X: sparse matrix of TFIDF scores or term frequencies

      Returns:
          M: dense numpy array of all pairwise cosine similarities.  That is, the 
             entry M[i,j], should correspond to the cosine similarity between the 
             ith and jth rows of X.
      """
    
      numdocs = X.get_shape()[0]
      M = X.copy().tocoo()
      # Power of 2
      M.data = M.data**2.0
      sumsq = np.array(M.sum(axis=1))
      # Power of 5
      rtsq = sumsq**.5
      invRtsq = 1.0/rtsq
      invRtsq_xpd = np.array([invRtsq[row][0] for row in M.row])
      M.data= M.data**.5*invRtsq_xpd

    
      return M.dot(M.transpose()).todense()
