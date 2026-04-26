import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

frases = [
    "oi", "olá", "bom dia",
    "que horas são", "me diga a hora",
    "qual seu nome", "quem é você",
    "me ensina python", "quero aprender programação"
]

classes = [
    "saudacao", "saudacao", "saudacao",
    "hora", "hora",
    "nome", "nome",
    "estudo", "estudo"
]

vec = CountVectorizer()
X = vec.fit_transform(frases)

modelo = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=3000)
modelo.fit(X, classes)

pickle.dump(modelo, open("modelo.pkl", "wb"))
pickle.dump(vec, open("vetorizador.pkl", "wb"))

print("Rede neural treinada!")