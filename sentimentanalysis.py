#!/usr/bin/env python
# Name: Charlie Yang
# Netid: cyang71
import nltk
import glob
import pickle
import string
from collections import defaultdict

# function used to get training data into binary list
def get_training_data(path, word_list, percentage):
	files = glob.glob(path)
	i = 0
	for file_name in files:
		with open(file_name) as f:
			if i < percentage * 10:
				for line in f:
					for word in nltk.word_tokenize(line):
						if word not in word_list:
							word_list.append(word)	
			i += 1

# function used to get training data into binary list of adjectives
def get_training_data_adjectives(path, word_list, percentage):
	files = glob.glob(path)
	adjective_token = {"JJ", "JJR", "JJS"}
	i = 0
	for file_name in files:
		with open(file_name) as f:
			if i < percentage * 10:
				for line in f:
					tokens = nltk.word_tokenize(line)
					for tagged in nltk.pos_tag(tokens):
						if tagged[1] in adjective_token and tagged[0] not in word_list:
							word_list.append(tagged[0])
			i += 1

# function used to test data on list and print out Tp and Tn values
def run_test(path, percentage):
	pos_count = 0
	neg_count = 0
	pos_review = 0
	neg_review = 0
	i = 0
	files = glob.glob(path)
	for file_name in files:
		with open(file_name) as f:
			if i >= percentage * 10:
				for line in f:
					for word in nltk.word_tokenize(line):
						if word in pos_list:
							pos_count += 1
						if word in neg_list:
							neg_count += 1
				if pos_count > neg_count:
					pos_review += 1
				else:
					neg_review += 1
				pos_count = 0
				neg_count = 0
			i += 1
	print("Positive:", pos_review / (pos_review + neg_review), "Negative:", neg_review / (pos_review + neg_review), "Percentage:", percentage)

# function used to test data on dictioanry and print out Tp and Tn values
def run_test_NRC(path):
	pos_count = 0
	neg_count = 0
	pos_review = 0
	neg_review = 0
	files = glob.glob(path)
	for file_name in files:
		with open(file_name) as f:
			for line in f:
				for word in nltk.word_tokenize(line):
					if "positive" in emotion_dict[word] and word in pos_list:
						pos_count += 1
					if "negative" in emotion_dict[word] and word in neg_list:
						neg_count += 1
			if pos_count > neg_count:
				pos_review += 1
			else:
				neg_review += 1
			pos_count = 0
			neg_count = 0
	print("Positive:", pos_review / (pos_review + neg_review), "Negative:", neg_review / (pos_review + neg_review))	

# declare lists
review_tuple = []
pos_list = []
neg_list = []

# Loop through first 700 texts of positive and negative reviews and get all words used
get_training_data("./review_polarity/txt_sentoken/pos/*.txt", pos_list, 70)
get_training_data("./review_polarity/txt_sentoken/neg/*.txt", neg_list, 70)

# Loop through remaining files and check if they contain more positive or negative words
print("Bag-of-word Approach")
run_test("./review_polarity/txt_sentoken/pos/*.txt", 70)
run_test("./review_polarity/txt_sentoken/neg/*.txt", 70)

# Clear lists
pos_list = []
neg_list = []

# # Loop through first 700 texts of positive and negative reviews and get all words used
get_training_data_adjectives("./review_polarity/txt_sentoken/pos/*.txt", pos_list, 70)
get_training_data_adjectives("./review_polarity/txt_sentoken/neg/*.txt", neg_list, 70)

# # Loop through remaining files and check if they contain more positive or negative adjectives
print("Bag-of-word Approach: Adjective")
run_test("./review_polarity/txt_sentoken/pos/*.txt", 70)
run_test("./review_polarity/txt_sentoken/neg/*.txt", 70)

# Go through the NRC emotion text file and create a dictionary associating each word with its emotion(s)
emotion_dict = defaultdict(list)
with open("NRC_Emotion.txt") as f:
	for line in f:
		word, emotion, value, *extrawords = line.split("\t")
		if value == "1\n":
			emotion_dict[word].append(emotion)

# Loop through remaining files and check if they contain more positive or negative words
print("Bag-of-word Approach: NRC")
run_test_NRC("./review_polarity/txt_sentoken/pos/*.txt")
run_test_NRC("./review_polarity/txt_sentoken/neg/*.txt")

# Import review tuples
print("Importing Reviews")
f = open('review_tuples.pickle', 'rb')
review_tuple = pickle.load(f)
f.close()

# Tokenize words in review that are adjectives
all_words = set(word.lower() for passage in review_tuple for word in nltk.word_tokenize(passage[0]) if "positive" in emotion_dict[word] or "negative" in emotion_dict[word])

# Import classifier
f = open('scikit_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

# Change reviews into a single string
files = glob.glob("./review_polarity/txt_sentoken/pos/*.txt")
i = 0
review = ""
review_list = []
for file_name in files:
	with open(file_name) as f:
		if i >= 700:
			review = " ".join(line.strip() for line in f)
			review_list.append(review)
		i += 1

# Begin classification
print("Beginning Classification")
pos_count = 0
neg_count = 0
for review in review_list:
	review = review.translate(str.maketrans('','',string.punctuation))
	review = " ".join(review.split())
	featurized_test_sentence =  {i:(i in nltk.word_tokenize(review.lower())) for i in all_words if "positive" in emotion_dict[i] or "negative" in emotion_dict[i]}
	verdict = classifier.classify(featurized_test_sentence)
	if verdict == "pos":
		pos_count += 1
	else:
		neg_count += 1
	print("Pos:", pos_count, "Neg:", neg_count)
pos_classify = pos_count / (pos_count + neg_count)

files = glob.glob("./review_polarity/txt_sentoken/neg/*.txt")
i = 0
review = ""
review_list = []
for file_name in files:
	with open(file_name) as f:
		if i >= 700:
			review = " ".join(line.strip() for line in f)
			review_list.append(review)
		i += 1

pos_count = 0
neg_count = 0
for review in review_list:
	review = review.translate(str.maketrans('','',string.punctuation))
	review = " ".join(review.split())
	featurized_test_sentence =  {i:(i in nltk.word_tokenize(review.lower())) for i in all_words if "positive" in emotion_dict[i] or "negative" in emotion_dict[i]}
	verdict = classifier.classify(featurized_test_sentence)
	if verdict == "pos":
		pos_count += 1
	else:
		neg_count += 1
	print("Pos:", pos_count, "Neg:", neg_count)

# Print Tp and Tn values
print(pos_classify, neg_count / (pos_count + neg_count))