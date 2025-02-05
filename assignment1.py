import re  # for preprocessing 
import collections  # for counting unigram & bigrams n-grams

class NGramModel:
    def __init__(self, train, val):
        self.train_set = train
        self.val_set = val
        self.bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.unigram_counts = collections.defaultdict(int)
        self.vocab = set()
    
    # code for preprocessing the text data (tokenization, lowercasing, punctuation removal, sentence boundaries)
    def preprocess(self, text):
        """Tokenize text and add sentence boundaries."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()  # Tokenize by whitespace
        tokens = ["<s>"] + tokens + ["</s>"]  # Add start and end tokens
        return tokens


    def train_bigram_model(self):
        """Train bigram model by computing bigram probabilities from the training text."""
        with open(self.train_set, "r", encoding="utf-8") as file:
            for line in file:
                tokens = self.preprocess(line)
                
                # Update unigram counts
                for token in tokens:
                    self.unigram_counts[token] += 1
                    self.vocab.add(token)
                
                # Update bigram counts
                for i in range(len(tokens) - 1):
                    self.bigram_counts[tokens[i]][tokens[i + 1]] += 1

    def get_bigram_probability(self, word1, word2):
        """Compute P(word2 | word1) = count(word1, word2) / count(word1)."""
        if self.unigram_counts[word1] == 0:
            return 0  # Avoid division by zero
        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]
    
    # code for unigram model
    def train_unigram_model(self):
        """Implement unigram probability model."""
        pass
    
    def get_unigram_probability(self, word):
        """Implement unigram probability calculation."""
        pass
    
    # code for smoothing
    def smoothing(self):
        """Implement Laplace smoothing."""
        pass
    
    # code for unknown word handling
    def unknown_word_handling(self):
        """Implement unknown word handling."""
        pass
    
    # code for perplexity calculation
    def calculate_perplexity(self):
        """Implement perplexity calculation."""
        pass

if __name__ == "__main__":
    train = "train.txt"
    val = "val.txt"
    
    model = NGramModel(train, val)
    model.train_bigram_model()
    for word1 in model.bigram_counts:
        for word2, count in model.bigram_counts[word1].items():
            if count > 2:  # Change threshold to adjust how common the pair should be
                print(f"'{word1}' -> '{word2}': {count}")
    # Test the bigram model
    print("Bigram Probability (example):", model.get_bigram_probability("cancellation", "policy"))
