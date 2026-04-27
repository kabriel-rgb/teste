import math
import random

# Funções Matemáticas
def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

# Dados do XOR (O que um neurônio sozinho não resolve)
entradas = [[0,0], [0,1], [1,0], [1,1]]
saidas_esperadas = [[0], [1], [1], [0]]

# Inicialização de Pesos e Bias aleatórios
# Camada Oculta (2 neurônios, cada um com 2 entradas)
pesos_ocultos = [[random.uniform(-1,1) for _ in range(2)] for _ in range(2)]
bias_oculto = [random.uniform(-1,1) for _ in range(2)]

# Camada de Saída (1 neurônio que recebe 2 entradas da camada oculta)
pesos_saida = [random.uniform(-1,1) for _ in range(2)]
bias_saida = random.uniform(-1,1)

taxa_aprendizado = 0.5

# Treinamento
for epoca in range(10000):
    for i in range(len(entradas)):
        # --- FORWARD (Passada para frente) ---
        # Neurônios Ocultos
        ativacao_oculta = []
        for j in range(2):
            soma = entradas[i][0]*pesos_ocultos[j][0] + entradas[i][1]*pesos_ocultos[j][1] + bias_oculto[j]
            ativacao_oculta.append(sigmoide(soma))
        
        # Neurônio de Saída
        soma_saida = ativacao_oculta[0]*pesos_saida[0] + ativacao_oculta[1]*pesos_saida[1] + bias_saida
        predicao = sigmoide(soma_saida)

        # --- BACKPROPAGATION (Ajuste dos pesos) ---
        # Erro na Saída
        erro_saida = saidas_esperadas[i][0] - predicao
        delta_saida = erro_saida * derivada_sigmoide(predicao)

        # Erro na Camada Oculta
        delta_oculto = []
        for j in range(2):
            erro_h = delta_saida * pesos_saida[j]
            delta_oculto.append(erro_h * derivada_sigmoide(ativacao_oculta[j]))

        # Atualização dos Pesos e Bias
        for j in range(2):
            pesos_saida[j] += ativacao_oculta[j] * delta_saida * taxa_aprendizado
            for k in range(2):
                pesos_ocultos[j][k] += entradas[i][k] * delta_oculto[j] * taxa_aprendizado
                
        bias_saida += delta_saida * taxa_aprendizado
        bias_oculto[0] += delta_oculto[0] * taxa_aprendizado
        bias_oculto[1] += delta_oculto[1] * taxa_aprendizado

# Teste Final
print("Resultados após treinar a rede:")
for i in range(len(entradas)):
    # Repete o Forward para testar
    h1 = sigmoide(entradas[i][0]*pesos_ocultos[0][0] + entradas[i][1]*pesos_ocultos[0][1] + bias_oculto[0])
    h2 = sigmoide(entradas[i][0]*pesos_ocultos[1][0] + entradas[i][1]*pesos_ocultos[1][1] + bias_oculto[1])
    res = sigmoide(h1*pesos_saida[0] + h2*pesos_saida[1] + bias_saida)
    print(f"Entrada: {entradas[i]} | Saída: {round(res, 2)}")
