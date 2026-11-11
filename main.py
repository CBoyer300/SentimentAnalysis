import os
import re
import xml.etree.ElementTree as et
import random
from collections import Counter

#Hyperparameters
k = 5 # k most frequent words to keep
maxReviewLength = 1200 #Most amount of words in a review to process

#Gets all subdirectories in a folder
def list_folders(directory):
    folders = [entry.path for entry in os.scandir(directory) if entry.is_dir()]

    pos = [folder + '/positive.review' for folder in folders]
    neg = [folder + '/negative.review' for folder in folders]

    return pos, neg

#Get stop words
stopWords = ''
with open('sorted_data/stopwords', 'r') as f:
    stopWords = f.read()
    
stopWords = stopWords.replace('\n', ' ').split()

#Get all subdirectories for sorted_data, seperate and combined
posFolders, negFolders = list_folders("sorted_data")
allFolders = posFolders + negFolders

#To clean up invalid text
def clean_xml(text):
    # Escape unescaped ampersands
    text = re.sub(
        r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)',
        '&amp;',
        text
    )

    # Remove invalid XML 1.0 control characters
    text = ''.join(
        c for c in text
        if c in '\t\n\r'
        or '\u0020' <= c <= '\uD7FF'
        or '\uE000' <= c <= '\uFFFD'
        or '\U00010000' <= c <= '\U0010FFFF'
    )

    return text

#Vocabulary for tokenizer and reverse
#Set of words, counts up by 1
#Encoding = [vocab[word] for word in tokens]
#Keep k most frequent words, will run into misspellings and abrevs

#padding
#unknown words
#Beginning of seq
#end of seq
stoi = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3
}
itos = {
    0: "[PAD]",
    1: "[UNK]",
    2: "[BOS]",
    3: "[EOS]"
}

word_counts = Counter()

#Parse a file to get a list of all the reviews in a file and their ratings, ignore everything else for now
def parser():
    dataset = []

    def processFileType(sentiment):
        print(file)
        xml = et.XMLPullParser(events=("start", "end"))
        xml.feed("<reviews>")

        root = None

        def process_events():
            nonlocal root

            for event, elem in xml.read_events():
                if event == "start" and elem.tag == "reviews":
                    root = elem

                elif event == "end" and elem.tag == "review":
                    rating = float(elem.find("rating").text.strip())
                    title = elem.find("title").text.strip()

                    #Get overall text and remove stopwords
                    text = elem.find("review_text").text.strip().replace('\n', ' ').lower().split()
                    text = [item for item in text if item not in stopWords]

                    #Count frequency of a given word
                    word_counts.update(text)

                    if len(text) <= maxReviewLength:
                        text = " ".join(text)
                        text = '[BOS] ' + text + ' [EOS]'
                        while len(text) < maxReviewLength:
                            text += ' [PAD]'
                        dataset.append({
                            "rating": rating,
                            "title": title,
                            "text": text,
                            "Sentiment": sentiment
                        })

                    root.remove(elem)
                    elem.clear()

        with open(file, "r") as f:
            while chunk := f.read(65536):
                xml.feed(clean_xml(chunk))
                process_events()

        xml.feed("</reviews>")
        process_events()
        xml.close()
        process_events()

    for file in posFolders:
            processFileType('positive')
    for file in negFolders:
        processFileType('negative')

    return dataset

#Dataset is array of dictionaries
#Rating: num, title: string, text: string, sentiment: string
dataset = parser()
random.shuffle(dataset)

#Adds k most frequent words to dataset, others become [UNK]
#encoding = .get(vocab[word], vocab[UNK]), gets the encoding for a word 
#or uses unknown if not found
stoi = {
    word: i + 4
    for i, (word, count) in enumerate(word_counts.items())
    if count >= k
}
itos = {
    i + 4: word
    for i, (word, count) in enumerate(word_counts.items())
    if count >= k
}

#Encode/decode a string of words
def encode(s):
    return [stoi.get(c, stoi["[UNK]"]) for c in s]

def decode(token):
    return ''.join(itos[i] for i in token)

print(len(dataset))
print(dataset[:10])

