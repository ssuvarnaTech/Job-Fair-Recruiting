import subprocess

import docx
import re
import os
import sys
import csv
from  pandas import DataFrame
from glob import glob
import re
import os

def extract_from_docx(file_name):

    doc = docx.Document(file_name)
    line = []
    splitBySentence = []
    emptyString = ""


    for i in doc.paragraphs:
       if i.text == '':
         continue
       sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', i.text)
       text = []
       for phrases in sentences:
           phrases =  re.split(r' *[\.\?!,;][\'"\)\]]* *', phrases)
           for modified_phrases in phrases:
            if modified_phrases == "":
                continue
            modified_phrases = modified_phrases.replace(",","")
            modified_phrases = modified_phrases.split(";")
            for s in modified_phrases:
                if s == "":
                    continue
                text.append(s)

       for j in text:
           splitBySentence.append(j)
    return splitBySentence


def extract_file_by_file(folder_path):
    sentences = []
    dirs = os.listdir(folder_path)
    for file in dirs:
       if(file == '.DS_Store'):
           continue
       dirs2 = os.listdir(folder_path + '/' + file)

       for f in dirs2:
           if(f.endswith('.docx')):
               sentences.append(extract_from_docx(folder_path + '/' + file + '/' + f))

    return sentences

def create_sample_dataset(file_path):
    lines = list()
    count = 0

    with open(file_path, 'r') as readFile:

        reader = csv.reader(readFile)

        for row in reader:
            if count == 2000:
                break
            lines.append(row)
            count+=1
            if row[0] in (None, ""):
                lines.remove(row)
                count-=1
    with open('mySample.csv', 'w') as writeFile:


        writer = csv.writer(writeFile)

        writer.writerows(lines)
#word embedding
def get_Sample_Word_Embedding(file):
    match_word_to_index = {}
    f = open(file, "r")
    count = 0
    for line in f:
        index_of_space = line.find("_")

        if count == 0:
            count+=1
            continue
        if count == 10000:
            break

        match_word_to_index[line[0:index_of_space:1]] = count
        count+=1
    f.close()

    return match_word_to_index
#put words in vector list
def get_Vectors(file):
    vectors = []
    f = open(file, "r")
    for line in f:
        lineSplit = line.split(" ")
        for x in lineSplit(2,len(lineSplit)):
            vectors.append(lineSplit[x])


    return vectors

    # for j in range(len(line)):
    #     word = line[j]
    #     if '.' in word:




def main():
   # print(extract_all_sentences('Resumes/1Amy.docx'))
   #  sentences = extract_file_by_file('Resumes/Original_Resumes')
   #  with open('test.csv', 'w') as f:
   #     writer = csv.writer(f)
   #     for s in sentences:
   #         for elements in s:
   #           writer.writerow([elements])
   #
   #  create_sample_dataset('test.csv')
   #get word embedding
    print(get_Sample_Word_Embedding("/Users/sreevanisuvarna/Downloads/3/model.txt"))
    print(get_Vectors("/Users/sreevanisuvarna/Downloads/3/model.txt"))










if __name__ == "__main__":
    main()