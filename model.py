# Converted from model2.ipynb
# Markdown cells are preserved as comments; code cells are kept in order.

# ========================================================================
# # NLP Model Lab Exam: Simple General Template
#
# **Before running:** install the required libraries and spaCy English model.
#
# ```bash
# pip install nltk spacy scikit-learn gensim transformers torch
# python -m spacy download en_core_web_sm
# ```
#
# The first NLTK / BERT run may need internet access to download resources and pretrained weights.
#
# ========================================================================

# ========================================================================
# ## Cell 1: Imports and load the CSV
# ========================================================================

import re, math, random
from collections import Counter

import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download(["punkt", "punkt_tab", "averaged_perceptron_tagger",
               "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"], quiet=True)
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("data.csv")       # Change to your CSV filename
print(df.head())
df.info()
print(df.describe(include="all"))
print("Columns:", df.columns.tolist())

TEXT_COL = "text"                 # Change to your actual text column
texts = df[TEXT_COL].dropna().astype(str).tolist()
if not texts:
    raise ValueError("No text found. Check TEXT_COL and the CSV file.")
print("Number of texts:", len(texts))

# ========================================================================
# ## Unit (a) - Cell 2: Preprocessing, stemming and lemmatization
# ========================================================================

def clean_tokens(raw_text):
    text = str(raw_text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Remove URLs

    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "i'm": "i am", "it's": "it is", "isn't": "is not",
        "didn't": "did not", "you're": "you are", "we're": "we are"
    }
    for old, new in contractions.items():
        text = text.replace(old, new)

    text = text.replace(":)", " <happy> ").replace(":(", " <sad> ")
    return re.findall(r"<happy>|<sad>|[a-z]+", text)  # Regex tokenization

sample = "I don't like this http://test.com :( I'm happy :)"
tokens = clean_tokens(sample)
print("Tokens:", tokens)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def wordnet_pos(tag):
    if tag.startswith("J"): return "a"
    if tag.startswith("V"): return "v"
    if tag.startswith("R"): return "r"
    return "n"

pos_tags = nltk.pos_tag(tokens)
lemmas = [lemmatizer.lemmatize(w, wordnet_pos(tag)) for w, tag in pos_tags]
print("Stems :", [stemmer.stem(w) for w in tokens])
print("Lemmas:", lemmas)

# ========================================================================
# ## Unit (a) - Cell 3: Bigram language model, smoothing, probability, perplexity and sampling
# ========================================================================

sentences = []
for text in texts:
    words = clean_tokens(text)
    if words:
        sentences.append(["<s>"] + words + ["</s>"])

unigrams = Counter(w for sent in sentences for w in sent)
bigrams = Counter()
context_counts = Counter()

for sent in sentences:
    for w1, w2 in zip(sent, sent[1:]):
        bigrams[(w1, w2)] += 1
        context_counts[w1] += 1

V = len(unigrams)

def bigram_prob(w1, w2):
    return (bigrams[(w1, w2)] + 1) / (context_counts[w1] + V)

# Probability and perplexity of the first sentence
test_sent = sentences[0]
log_prob = sum(math.log(bigram_prob(a, b)) for a, b in zip(test_sent, test_sent[1:]))
N = len(test_sent) - 1
print("Log probability:", log_prob)
print("Sentence probability:", math.exp(log_prob))
print("Perplexity:", math.exp(-log_prob / N))

# Random sentence sampling with a fixed seed
random.seed(42)
vocab = [w for w in unigrams if w != "<s>"]
word = "<s>"
sampled = []
for _ in range(20):
    word = random.choices(vocab, weights=[bigram_prob(word, w) for w in vocab])[0]
    if word == "</s>":
        break
    sampled.append(word)
print("Sampled sentence:", " ".join(sampled))

# ========================================================================
# ## Unit (b) - Cell 4: POS tagging and open-class / closed-class words
# ========================================================================

tokens = nltk.word_tokenize(texts[0])
tagged = nltk.pos_tag(tokens)

open_class = [w for w, tag in tagged if tag.startswith(("NN", "VB", "JJ", "RB"))]
closed_class = [w for w, tag in tagged if tag.startswith(("DT", "IN", "PRP", "CC"))]

print("POS tags:", tagged)
print("Open-class words:", open_class)
print("Closed-class words:", closed_class)

# ========================================================================
# ## Unit (b) - Cell 5: CFG parsing and PP-attachment ambiguity (two trees)
# ========================================================================

grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP | VP PP
NP -> DT N | NP PP | 'i'
PP -> P NP
V -> 'saw'
DT -> 'the'
N -> 'man' | 'telescope'
P -> 'with'
""")

parser = nltk.ChartParser(grammar)
sentence = "i saw the man with the telescope".split()
trees = list(parser.parse(sentence))
print("Number of parse trees:", len(trees))
for tree in trees:
    tree.pretty_print()

# ========================================================================
# ## Unit (b) - Cell 6: spaCy dependencies and aspect-opinion pairs
# ========================================================================

doc = nlp("The battery is amazing and the phone has a great screen.")

for token in doc:
    print(token.text, "--", token.dep_, "--> ", token.head.text)

pairs = []
for token in doc:
    # Adjective modifies a noun: great -> screen
    if token.dep_ == "amod":
        pairs.append((token.head.text, token.text))

    # Adjectival complement: battery is amazing
    if token.dep_ == "acomp":
        subject = next((child.text for child in token.head.children
                        if child.dep_ == "nsubj"), None)
        if subject:
            pairs.append((subject, token.text))

print("Aspect-opinion pairs:", pairs)

# ========================================================================
# ## Unit (c) - Cell 7: WordNet synsets, Wu-Palmer similarity and Lesk WSD
# ========================================================================

word1, word2 = "bank", "money"
syn1, syn2 = wn.synsets(word1), wn.synsets(word2)
print("Synsets:", syn1)

if syn1 and syn2:
    print("Wu-Palmer similarity (first senses):", syn1[0].wup_similarity(syn2[0]))

context = nltk.word_tokenize("He went to the bank to deposit his money.")
sense = lesk(context, "bank")
print("Lesk sense:", sense)
print("Definition:", sense.definition() if sense else "No sense found")

# ========================================================================
# ## Unit (c) - Cell 8: TF-IDF, cosine similarity and top-k retrieval
# ========================================================================

tfidf = TfidfVectorizer()
doc_matrix = tfidf.fit_transform(texts)

query_text = " " + texts[0]  # Replace with any search query
query_vector = tfidf.transform([query_text])
scores = cosine_similarity(query_vector, doc_matrix).flatten()

k = min(3, len(texts))
top_k = scores.argsort()[::-1][:k]
for i in top_k:
    print("Score:", round(scores[i], 3), "| Text:", texts[i])

# ========================================================================
# ## Unit (c) - Cell 9: PPMI with window ±2 and Word2Vec Skip-Gram
# ========================================================================

tokenized = [clean_tokens(t) for t in texts]

# Keep the 500 most frequent words so the dense matrix fits lab computers
word_counts = Counter(w for sent in tokenized for w in sent)
vocab = [w for w, count in word_counts.most_common(500)]
word_to_id = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))

# Co-occurrence counts in a window of 2 words on each side
for sent in tokenized:
    for i, word in enumerate(sent):
        if word not in word_to_id:
            continue
        for j in range(max(0, i - 2), min(len(sent), i + 3)):
            other = sent[j]
            if i != j and other in word_to_id:
                C[word_to_id[word], word_to_id[other]] += 1

row_sum = C.sum(axis=1, keepdims=True)
col_sum = C.sum(axis=0, keepdims=True)
ppmi = np.maximum(0, np.log2((C * C.sum() + 1e-9) / (row_sum @ col_sum + 1e-9)))
print("PPMI sample:\n", ppmi[:4, :4].round(2))

# sg=1 means Skip-Gram; workers=1 and seed improve reproducibility
w2v = Word2Vec(sentences=tokenized, vector_size=50, window=2,
               min_count=1, sg=1, seed=42, workers=1)
if len(w2v.wv) > 1:
    print("Similar words:", w2v.wv.most_similar(vocab[0], topn=3))

# ========================================================================
# ## Unit (d) - Cell 10: Mentions, pair scoring and simple coreference chains
# ========================================================================

doc = nlp("Mr. John bought a laptop. The man liked the laptop. He used it daily.")

names = [e for e in doc.ents if e.label_ == "PERSON"]
chunks = list(doc.noun_chunks)
pronouns = [doc[t.i:t.i + 1] for t in doc if t.pos_ == "PRON"]
print("Names:", [m.text for m in names])
print("Noun chunks:", [m.text for m in chunks])
print("Pronouns:", [m.text for m in pronouns])

# Combine spans and remove duplicates
unique = {(m.start, m.end): m for m in names + chunks + pronouns}
mentions = sorted(unique.values(), key=lambda m: m.start)

male = {"mr", "john", "rahul", "he", "him", "his", "man", "boy"}
female = {"mrs", "mary", "priya", "she", "her", "woman", "girl"}
plural = {"they", "them", "their", "people"}
pronoun_words = {"he", "him", "his", "she", "her", "they", "them", "their", "it", "its"}

# Make a sentence number for each token
sentence_number = {}
for i, sent in enumerate(doc.sents):
    for token in sent:
        sentence_number[token.i] = i

def gender_number(m):
    words = set(re.findall(r"[a-z]+", m.text.lower()))
    gender = "M" if words & male else ("F" if words & female else "N")
    number = "P" if words & plural or m.root.tag_ in {"NNS", "NNPS"} else "S"
    return gender, number

def pair_score(old, new):
    g1, n1 = gender_number(old)
    g2, n2 = gender_number(new)
    if n1 != n2 or (g1 != g2 and (g1 != "N" or g2 != "N")):
        return -10

    same_head = old.root.lemma_.lower() == new.root.lemma_.lower()
    is_pronoun = new.text.lower().strip(" .") in pronoun_words
    if not same_head and not is_pronoun:
        return -10

    score = 2  # Number agreement
    if g1 == g2 and g1 != "N":
        score += 2  # Known gender agreement
    if same_head:
        score += 2  # Head-word match
    distance = sentence_number[new.root.i] - sentence_number[old.root.i]
    return score - distance

chains = []
mention_chain = []
for i, mention in enumerate(mentions):
    best_j, best_score = -1, 0
    for j in range(i):
        score = pair_score(mentions[j], mention)
        if score > best_score:
            best_j, best_score = j, score

    if best_j == -1:
        chains.append([mention.text])
        mention_chain.append(len(chains) - 1)
    else:
        chain_id = mention_chain[best_j]
        chains[chain_id].append(mention.text)
        mention_chain.append(chain_id)

print("Coreference chains:", [c for c in chains if len(c) > 1])

# ========================================================================
# ## Unit (e) - Cell 11: BERT fill-mask and sentiment pipelines
# ========================================================================

fill_mask = pipeline("fill-mask", model="bert-base-uncased")
print("Fill-mask:", fill_mask("NLP is [MASK].", top_k=3))

sentiment = pipeline("sentiment-analysis")
print("Sentiment:", sentiment("This course is helpful and interesting!"))

# ========================================================================
# ## Unit (e) - Cell 12: Fine-tune BERT on 9 samples, accuracy, F1 and new prediction
# ========================================================================

train_texts = [
    "I love this product", "Amazing quality and service", "Great experience overall",
    "Very happy with purchase", "Best item I bought", "Terrible quality and bad",
    "I hate this product", "Worst experience ever", "Broken and useless"
]
train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0]  # 1=positive, 0=negative

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

inputs = tokenizer(train_texts, padding=True, truncation=True,
                   max_length=64, return_tensors="pt")
labels = torch.tensor(train_labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch + 1, "Loss:", round(loss.item(), 4))

# These are training-set scores because the exercise uses only 9 samples
model.eval()
with torch.no_grad():
    predictions = model(**inputs).logits.argmax(dim=1).numpy()
print("Training accuracy:", accuracy_score(train_labels, predictions))
print("Training F1:", f1_score(train_labels, predictions, zero_division=0))

new_text = "This product is wonderful and useful."
new_inputs = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    new_pred = model(**new_inputs).logits.argmax(dim=1).item()
print("New prediction:", "Positive" if new_pred == 1 else "Negative")
