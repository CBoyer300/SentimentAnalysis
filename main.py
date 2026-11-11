import os
import re
import xml.etree.ElementTree as et

#Gets all subdirectories in a folder
def list_folders(directory):
    folders = [entry.path for entry in os.scandir(directory) if entry.is_dir()]

    pos = [folder + '/positive.review' for folder in folders]
    neg = [folder + '/negative.review' for folder in folders]

    return pos, neg

#Get all subdirectories for sorted_data, seperate and combined
posFolders, negFolders = list_folders("sorted_data")
allFolders = posFolders + negFolders

#To clean up ampersands
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

#Parse a file to get a list of all the reviews in a file and their ratings, ignore everything else for now
def parser():
    dataset = []

    for file in allFolders:
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
                    text = elem.find("review_text").text.strip()

                    dataset.append({
                        "rating": rating,
                        "title": title,
                        "text": text
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

    return dataset
         
dataset = parser()
print(dataset[:5])
print(len(dataset))