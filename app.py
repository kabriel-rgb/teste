from flask import Flask, render_template, request, jsonify
import pickle
import datetime

app = Flask(__name__)

modelo = pickle.load(open("modelo.pkl", "rb"))
vec = pickle.load(open("vetorizador.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["msg"]

    entrada = vec.transform([msg])
    classe = modelo.predict(entrada)[0]

    if classe == "saudacao":
        resp = "Olá user. Sistemas online."
    elif classe == "hora":
        resp = "Agora são " + datetime.datetime.now().strftime("%H:%M:%S")
    elif classe == "nome":
        resp = "Eu sou SAHUR."
    elif classe == "estudo":
        resp = "Python é excelente para começar."
    else:
        resp = "Ainda estou aprendendo."

    return jsonify({"resposta": resp})

app.run(debug=True)