----------------------------------------
Name: Charlie Yang
Netid: cyang71
----------------------------------------
The programs are written and tested for Python 3.4 using Windows 10 Bash
----------------------------------------
PYTHON FILES:

How naivebayes.py, decisiontree.py, scikit.py works (~15 hours run-time total):
RECOMMENDED YOU DO NOT RUN UNLESS NECSSARY. EACH PROGRAM TAKES ~5 HOURS
Each program takes in the NRC_Emotion.txt file and scans through the file to create
a dictionary mapping a word to a list of emotions associated with it. It then scans
through the movies reviews and adds them to a list of tuples of (review, sentiment).
Then it uses NLTK to tokenize each review and if the token exists in the emotion
dictionary, it adds it to the feature set. The feature set is then sent through
either the Naive-Bayes, Decision Tree, or Support Vector Machine to train the
classifier. The classifier is then stored into a pickle file that can be accessed
from sentimentanalysis.py.

How sentimentanalysis.py works (~4 hours run-time total):
The program first runs the baseline system using the bag-of-words approach by splitting
the data into testing (default 70%) and training (default 30%) data. It then runs the
baseline model to assess the performance metrics. 
The program then introduces features to the system by only taking in adjectives as defined
by the Penn-Treebank part-of-speech tagging. It then runs the model to assess the performance
metrics.
The program then uses the NRC_Emotion.txt to create a dictionary of words and a list of their
associated emotions. It then runs the model to assess the performance metric.
It then imports the pickle file (defaut Naive-Bayes) for the supervised machine learning
models and imports the list of reviews (review_tuples.pickle). It then runs the model to
assess the performance metrics.
----------------------------------------
OTHER FILES:

review_polarity: 
folder containing 1000 positive reviews and 1000 negative reviews used for testing
and training.

decisiontree_classifier, naivebayes_classifier, scikit_classifier:
pickle files containing data created by their respective .py files used to speed up
computation.

review_tuples:
pickle file containing a list of reviews and their sentiment used to speed up
computation.

NRC_Emotion.txt:
text file containing a list of words, emotions, and whether a word and an emotion
are related.
----------------------------------------
How to run the program (Windows 10):
1) Head to https://www.python.org/downloads/release/python-344/ scroll to the bottom of 
the page to download the installer
2) Click next until the installer finishes, then click finish
3) Press (Win X) followed by (Y) to get to the System page
4) On the left hand side, press "Advanced System Settings"
5) A window should pop up and on the bottom right, press "Environmental Variables"
6) Under "System Variables" locate the variable named "PATH", click on it, and 
press the "Edit" button
7) Press browse and locate the folder "Python34" (default installation is C:\), 
click the folder, and press "OK"
8) Press "OK" to exit the Edit pop-up and then close everything else
9) Locate the folder containing the program and while holding "Shift" down, 
right-click the folder, and press "Open command window here"
10) Type "python3.4 program_name.py" and press Enter to run the program
----------------------------------------