class LanguageModel:
    def __init__(self, docs, n):
        """
        Initialize an n-gram language model.
        
        Args:
            docs: list of strings, where each string represents a space-separated
                  document
            n: integer, degree of n-gram model
        """
        count_sums = {}
        counts = {}
        for doc in docs:
            tokens = doc.split()
            prevN = []
            for i,token in enumerate(tokens[:-1]):
                if i==len(tokens)-1:
                    print ()
                prevN.append(token)
                if len(prevN)>=n-1:
                    prevNstr = " ".join(prevN)
                    if prevNstr in counts:
                        if tokens[i+1] in counts[prevNstr]:
                            counts[prevNstr][tokens[i+1]] += 1
                            count_sums[prevNstr] += 1
                        else:
                            counts[prevNstr][tokens[i+1]] = 1
                            count_sums[prevNstr] += 1
                    else:
                        counts[prevNstr] = {tokens[i+1]:1}
                        count_sums[prevNstr] = 1
                    prevN = prevN[1:]
                    
        self.count_sums = count_sums
        self.counts = counts
        self.dictionary = {word for doc in docs for word in doc.split()}
        self.n = n
    
    def perplexity(self, text, alpha=1e-3):
        """
        Evaluate perplexity of model on some text.
        
        Args:
            text: string containing space-separated words, on which to compute
            alpha: constant to use in Laplace smoothing
            
        Note: for the purposes of smoothing, the dictionary size (i.e, the D term)
        should be equal to the total number of unique words used to build the model
        _and_ in the input text to this function.
            
        Returns: perplexity
            perplexity: floating point value, perplexity of the text as evaluted
                        under the model.
        """
        tokens = text.split()
        D = len(set(tokens).union(self.dictionary))
        N = len(tokens)
        
        wordLogProbs = []
        for i,word in enumerate(tokens):
            num = alpha
            denom = alpha*D
            if i >= self.n - 1:
                prevN = tokens[(i-(self.n-1)):i]
                prevNstr = " ".join(prevN)
                if prevNstr in self.counts:
                    if word in self.counts[prevNstr]:
                        num += self.counts[prevNstr][word]
                    denom += self.count_sums[prevNstr]
                wordLogProbs.append(np.log(num)-np.log(denom))
            
        perp = np.exp(-1.0/(N-self.n+1) *(sum(wordLogProbs)))
        return perp
        
    def sample(self, k):
        """
        Generate a random sample of k words.
        
        Args:
            k: integer, indicating the number of words to sample
            
        Returns: text
            text: string of words generated from the model.
        """
        def getKeyListProp(d):
            key_lst_lst = [[key]*count for key,count in d.items()]
            key_lst = [key for key_lst in key_lst_lst for key in key_lst]
            return key_lst        
        def getRandKey(key_lst):
            rand_index = np.random.randint(len(key_lst))
            return key_lst[rand_index]
        
        count_sumsKeyList = getKeyListProp(self.count_sums)
        
        words = []
        prevN = getRandKey(count_sumsKeyList)
        words.extend(prevN.split())
        while len(words)<k:
            if len(words)>=self.n-1:
                prevN = " ".join(words[-(self.n-1):])
                if prevN in self.counts:
                    prevNdict = self.counts[prevN]
                else:
                    prevN = getRandKey(count_sumsKeyList)
                    prevNdict = self.counts[prevN]
            else:
                prevN = getRandKey(count_sumsKeyList)
                prevNdict = self.counts[prevN]
                
            lastN = getRandKey(getKeyListProp(prevNdict))
            
            words.append(lastN)
        text = " ".join(words)
        return text
