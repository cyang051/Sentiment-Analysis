import nltk
import glob
import pickle
from collections import defaultdict

emotion_dict = defaultdict(list)
with open("NRC_Emotion.txt") as f:
	for line in f:
		word, emotion, value, *extrawords = line.split("\t")
		if value == "1\n":
			emotion_dict[word].append(emotion)

files = glob.glob("./review_polarity/txt_sentoken/pos/*.txt")
i = 0
review = ""
review_tuple = []
for file_name in files:
	with open(file_name) as f:
		if i < 700:
			review = " ".join(line.strip() for line in f)
			review_tuple.append((review, "pos"))
		i += 1

files = glob.glob("./review_polarity/txt_sentoken/neg/*.txt")
i = 0
for file_name in files:
	with open(file_name) as f:
		if i < 700:
			review = " ".join(line.strip() for line in f)
			review_tuple.append((review, "neg"))
		i += 1

all_words = set(word.lower() for passage in review_tuple for word in nltk.word_tokenize(passage[0]))
t = [({word: (word in nltk.word_tokenize(x[0])) for word in all_words if "positive" in emotion_dict[word] or "negative" in emotion_dict[word]}, x[1]) for x in review_tuple]
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

f = open('naivebayes_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()