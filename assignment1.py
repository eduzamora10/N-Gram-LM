import re  # for preprocessing
import collections  # for counting unigram & bigrams n-grams
import math #for logarithms in perplexity calculation

class NGramModel:
    def __init__(self, train, val):
        self.train_set = train
        self.val_set = val
        self.bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.unigram_counts = collections.defaultdict(int)
        self.raw_bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))  # Store raw bigrams
        self.raw_unigram_counts = collections.defaultdict(int)  # Store raw unigrams
        self.vocab = set()

        # Train model upon initialization
        self.train_ngram_model()

    # code for preprocessing the text data (tokenization, lowercasing, punctuation removal, sentence boundaries)
    def preprocess(self, text):
        """Tokenize text and add sentence boundaries."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = text.split()  # Tokenize by whitespace
        tokens = ["<s>"] + tokens + ["</s>"]  # Add start and end tokens
        return tokens
    
    def train_ngram_model(self):
        """Train bigram model (and implicitly unigram model) by computing bigram probabilities from the training text."""
        
        with open(self.train_set, "r", encoding="utf-8") as file:
            for line in file:
                tokens = self.preprocess(line)

                # Update raw unigram counts
                for token in tokens:
                    self.raw_unigram_counts[token] += 1
                    self.vocab.add(token)

                # Update raw bigram counts
                for i in range(len(tokens) - 1):
                    self.raw_bigram_counts[tokens[i]][tokens[i + 1]] += 1

        # Copy raw counts before unknown word handling
        self.unigram_counts = self.raw_unigram_counts.copy()
        self.bigram_counts = self.raw_bigram_counts.copy()

        # Handle unknown words after counting
        self.unknown_word_handling()

    # get the bigram probabilit after unknown word handling
    def get_bigram_probability(self, word1, word2):
        """Compute P(word2 | word1) = count(word1, word2) / count(word1)."""
        if word1 not in self.vocab:
            word1 = "<UNK>"
        if word2 not in self.vocab:
            word2 = "<UNK>"

        if self.unigram_counts[word1] == 0:
            return 0  # Avoid division by zero

        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]

    # get the unigram probability after unknown word handling
    def get_unigram_probability(self, word):
        """Compute P(word) = count(word) / total words."""
        if word not in self.vocab:
            word = "<UNK>"

        total_words = sum(self.unigram_counts.values())
        return self.unigram_counts[word] / total_words if total_words > 0 else 0

    # get the original probabilities before unknown word handling
    def get_raw_bigram_probability(self, word1, word2):
        """Retrieve the original bigram probability before unknown word handling."""
        if word1 not in self.raw_unigram_counts or word2 not in self.raw_unigram_counts:
            return 0

        return self.raw_bigram_counts[word1][word2] / self.raw_unigram_counts[word1]
    
    # get the original probabilities before unknown word handling
    def get_raw_unigram_probability(self, word):
        """Retrieve the original unigram probability before unknown word handling."""
        total_words = sum(self.raw_unigram_counts.values())
        return self.raw_unigram_counts[word] / total_words if total_words > 0 else 0
        
    # code for smoothing
    def laplace(self):
        """Implement Laplace smoothing."""
        pass

    def addK(self, k: int, ngram: str, word1: str, word2=None):
        """Implement Add-K smoothing."""
        if word1 not in self.vocab:
            word1 = "<UNK>"
        if word2 and word2 not in self.vocab:
            word2 = "<UNK>"

        total_words = sum(self.unigram_counts.values())
        v = len(self.vocab)

        if ngram == "unigram":
            return (self.unigram_counts[word1] + k) / (total_words + k * v)
        elif ngram == "bigram":
            return (self.bigram_counts[word1][word2] + k) / (self.unigram_counts[word1] + k * v)
        else:
            raise ValueError("That's not a supported ngram.")

    # code for unknown word handling
    def unknown_word_handling(self, threshold=1):
        """Replace rare words with <UNK>."""
        rare_words = {word for word, count in self.unigram_counts.items() if count <= threshold}

        # Rebuild unigram counts with <UNK>
        new_unigram_counts = collections.defaultdict(int)
        for word, count in self.unigram_counts.items():
            if word in rare_words:
                new_unigram_counts["<UNK>"] += count  # Merge rare words into <UNK>
            else:
                new_unigram_counts[word] = count  # Keep frequent words unchanged

        # Rebuild bigram counts with <UNK>
        new_bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for w1 in self.bigram_counts:
            for w2 in self.bigram_counts[w1]:
                new_w1 = "<UNK>" if w1 in rare_words else w1
                new_w2 = "<UNK>" if w2 in rare_words else w2
                new_bigram_counts[new_w1][new_w2] += self.bigram_counts[w1][w2]

        self.unigram_counts = new_unigram_counts
        self.bigram_counts = new_bigram_counts
        self.vocab = set(self.unigram_counts.keys())  # Update vocab

    # code for perplexity calculation
    def calculate_perplexity(self):
        """Implement perplexity calculation."""
        total_log_prob = 0
        total_words = 0
        
        with open(self.val_set, "r", encoding="utf-8") as file:
            for line in file:
                tokens = self.preprocess(line)
                total_words += len(tokens) - 1  # Excluding the start token
                
                for i in range(len(tokens) - 1):
                    word1, word2 = tokens[i], tokens[i + 1]
                    prob = self.get_bigram_probability(word1, word2)
                    
                    if prob > 0:
                        total_log_prob += math.log(prob)
                    else:
                        total_log_prob += math.log(1e-10)  # Avoid log(0) by using a small probability
        
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity
    
if __name__ == "__main__":
    train = "train.txt"
    val = "val.txt"

    model = NGramModel(train, val)
    # for word1 in model.bigram_counts:
    #     for word2, count in model.bigram_counts[word1].items():
    #         if count > 2:  # Change threshold to adjust how common the pair s"hould be
    #             print(f"'{word1}' -> '{word2}': {count}")
    # Example with a known and unknown word
    print("Bigram Probability (handling unknown-words):", model.get_bigram_probability("xyz", "to"))
    print("Unigram Probability (handling unknown-words):", model.get_unigram_probability("xyz"))
    print("Bigram Probability (example, add-k)", model.addK(0.05, "bigram", "xyz", "to"))
    # Retrieve the original probabilities before unknown word handling
    print("Bigram Probability (unsmoothed):", model.get_raw_bigram_probability("xyz", "to"))
    print("Unigram Probability (unsmoothed):", model.get_raw_unigram_probability("xyz"))

    #compute and print perplexity 
    print("Perplexity:", model.calculate_perplexity())
