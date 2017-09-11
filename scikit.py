import nltk
import glob
import pickle
from collections import defaultdict
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

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

classifier = nltk.SklearnClassifier(LinearSVC()).train(t)

review = "films adapted from comic books have had plenty of success whether they re about superheroes batman superman spawn or geared toward kids casper or the arthouse crowd ghost world but there s never really been a comic book like from hell before for starters it was created by alan moore and eddie campbell who brought the medium to a whole new level in the mid 80s with a 12 part series called the watchmen to say moore and campbell thoroughly researched the subject of jack the ripper would be like saying michael jackson is starting to look a little odd the book or graphic novel if you will is over 500 pages long and includes nearly 30 more that consist of nothing but footnotes in other words don t dismiss this film because of its source if you can get past the whole comic book thing you might find another stumbling block in from hell s directors albert and allen hughes getting the hughes brothers to direct this seems almost as ludicrous as casting carrot top in well anything but riddle me this who better to direct a film that s set in the ghetto and features really violent street crime than the mad geniuses behind menace ii society the ghetto in question is of course whitechapel in 1888 london s east end it s a filthy sooty place where the whores called unfortunates are starting to get a little nervous about this mysterious psychopath who has been carving through their profession with surgical precision when the first stiff turns up copper peter godley robbie coltrane the world is not enough calls in inspector frederick abberline johnny depp blow to crack the case abberline a widower has prophetic dreams he unsuccessfully tries to quell with copious amounts of absinthe and opium upon arriving in whitechapel he befriends an unfortunate named mary kelly heather graham say it isn t so and proceeds to investigate the horribly gruesome crimes that even the police surgeon can t stomach i don t think anyone needs to be briefed on jack the ripper so i won t go into the particulars here other than to say moore and campbell have a unique and interesting theory about both the identity of the killer and the reasons he chooses to slay in the comic they don t bother cloaking the identity of the ripper but screenwriters terry hayes vertical limit and rafael yglesias les mis rables do a good job of keeping him hidden from viewers until the very end it s funny to watch the locals blindly point the finger of blame at jews and indians because after all an englishman could never be capable of committing such ghastly acts and from hell s ending had me whistling the stonecutters song from the simpsons for days who holds back the electric car who made steve guttenberg a star don t worry it ll all make sense when you see it now onto from hell s appearance it s certainly dark and bleak enough and it s surprising to see how much more it looks like a tim burton film than planet of the apes did at times it seems like sleepy hollow 2 the print i saw wasn t completely finished both color and music had not been finalized so no comments about marilyn manson but cinematographer peter deming don t say a word ably captures the dreariness of victorian era london and helped make the flashy killing scenes remind me of the crazy flashbacks in twin peaks even though the violence in the film pales in comparison to that in the black and white comic oscar winner martin childs shakespeare in love production design turns the original prague surroundings into one creepy place even the acting in from hell is solid with the dreamy depp turning in a typically strong performance and deftly handling a british accent ians holm joe gould s secret and richardson 102 dalmatians log in great supporting roles but the big surprise here is graham i cringed the first time she opened her mouth imagining her attempt at an irish accent but it actually wasn t half bad the film however is all good 2 00 r for strong violence gore sexuality language and drug content"
featurized_test_sentence =  {i:(i in nltk.word_tokenize(review.lower())) for i in all_words if "positive" in emotion_dict[i] or "negative" in emotion_dict[i]}
print(classifier.classify(featurized_test_sentence))

f = open('scikit_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()