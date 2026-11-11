import os
import random as r

#Gets all subdirectories in a folder
def list_folders(directory):
   return [directory + '/' + entry.name + '/positive.review' for entry in os.scandir(directory) if entry.is_dir()], [directory + '/' + entry.name + '/negative.review' for entry in os.scandir(directory) if entry.is_dir()]

#Get all subdirectories for sorted_data, seperate and combined
posFolders, negFolders = list_folders("sorted_data")
allFolders = posFolders + negFolders

#Parse a file to get a list of all the reviews in a file and their ratings, ignore everything else for now
def parser(path):
   pass