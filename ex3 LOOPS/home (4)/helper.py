import pickle

with open('dictionary.pkl', 'rb') as f:
    student_dict = pickle.load(f)
    
students_list = list(student_dict.keys())
scores_list = list(student_dict.values())