#Importing libraries
import pickle

#Using pickle to load the books_list
with open('books_dictionary.pkl', 'rb') as f:
    books_dict= pickle.load(f)
    
authors_list = list(books_dict.values())
books_list = list(books_dict.keys())


