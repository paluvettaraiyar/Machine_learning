import pickle

load_model = pickle.load(open('finalized_model.sav', 'rb'))

print(load_model.predict([[327.0, 119.0, 4.1, 4.4, 4.5, 9.50, 1]]))
