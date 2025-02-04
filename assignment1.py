class NGramModel:
    def __init__(self, train, val):
        self.train_set = train
        self.val_set = val
        pass
    
    # code for unigram model
    def unigram_model():
        pass
    
    # code for bigram model
    def bigram_model():
        pass
        
    # code for smoothing (laplace)
    def smoothing():
        pass
    
    # code for unknown word handling
    def unknown_word_handling():
        pass
    
    # code for calculating perplexity
    def calculate_perplexity():
        pass

if __name__ == "__main__":
    train = "train.txt"
    val = "val.txt"
    
    model = NGramModel(train, val)
    
    # call models and evaluation functions