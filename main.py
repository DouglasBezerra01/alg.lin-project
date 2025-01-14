import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import deque
import json

def gerar_dados(n):
    # Função para gerar matriz e vetor aleatórios.
    A = np.random.randint(1, 10, (n, n))  # Matriz A
    b = np.random.randint(1, 10, n)       # Vetor b
    return A, b

def resolver_sistema(A, b):
    # Função para resolver o sistema linear usando decomposição LU.
    P, L, U = scipy.linalg.lu(A)  # Decomposição LU
    y = scipy.linalg.solve(L, b)  # Resolução de Ly = b
    x = scipy.linalg.solve(U, y)  # Resolução de Ux = y
    return x

def plotar_resultados(A, b, x):
    # Função para plotar os resultados.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    ax.bar(range(len(x)), x, label='Solução x', color=cmap(0))
    ax.plot(range(len(A)), b, label='Vetor b', color=cmap(1), linestyle='--', marker='o')

    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor')
    ax.set_title('Solução do Sistema Linear e Resultado b')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    for i, (xi, bi) in enumerate(zip(x, b)):
        ax.annotate(f'{xi:.2f}', (i, xi), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        ax.annotate(f'{bi:.2f}', (i, bi), textcoords="offset points", xytext=(0,-15), ha='center', color='red')

    plt.savefig("resultado_grafico.png", dpi=300)
    plt.show()
    print("Gráfico salvo como 'resultado_grafico.png'.")

def armazenar_em_fila(x):
    # Função para armazenar a solução em uma fila.
    fila = deque(np.float64(x))
    return fila

def validar_entrada():
    # Função para validar a entrada do usuário.
    while True:
        try:
            n = int(input("Digite o tamanho da matriz (n): "))  # Tamanho da matriz
            if n > 0:
                return n
            else:
                print("Por favor, insira um número maior que zero.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número inteiro.")

def entrada_manual(n):
    # Função para permitir a entrada manual de A e b.
    A = np.zeros((n, n), dtype=int)
    b = np.zeros(n, dtype=int)
    print("Digite os elementos da matriz A:")
    for i in range(n):
        for j in range(n):
            while True:
                try:
                    A[i, j] = int(input(f"A[{i}][{j}] = "))
                    break
                except ValueError:
                    print("Entrada inválida. Por favor, insira um número inteiro.")
    print("Digite os elementos do vetor b:")
    for i in range(n):
        while True:
            try:
                b[i] = int(input(f"b[{i}] = "))
                break
            except ValueError:
                print("Entrada inválida. Por favor, insira um número inteiro.")
    return A, b

def salvar_resultados_em_json(A, b, x, filename="resultados.json"):
    # Função para salvar os resultados em um arquivo JSON.
    resultados = {
        "Matriz A": A.tolist(),
        "Vetor b": b.tolist(),
        "Solução x": x.tolist(),
    }
    with open(filename, "w") as f:
        json.dump(resultados, f, indent=4)
    print(f"Resultados salvos em '{filename}'.")

def main():
    # Entradas
    n = validar_entrada()

    # Escolha entre gerar dados aleatórios ou entrada manual
    escolha = input("Escolha como deseja inserir os dados: \n"
                    "1 - Gerar dados aleatórios\n"
                    "2 - Inserir manualmente\n"
                    "Digite sua escolha [1/2]: ").strip()
    if escolha == '2':
        A, b = entrada_manual(n)
    else:
        A, b = gerar_dados(n)

    # Verificar se a matriz A é invertível e bem condicionada
    if np.linalg.det(A) == 0 or np.linalg.cond(A) > 1e10:
        print("A matriz A não é invertível ou é mal condicionada. Por favor, insira outra matriz.")
        return

    # Resolução do sistema
    x = resolver_sistema(A, b)

    # Armazenando solução em uma fila
    fila_solucoes = armazenar_em_fila(x)
    print("\nFila com a solução do sistema x:")
    print(f"deque([{', '.join(f'{v:.2f}' for v in fila_solucoes)}])")

    # Visualização
    plotar_resultados(A, b, x)

    # Salvar os resultados em um arquivo JSON
    salvar_resultados_em_json(A, b, x)

    # Exibição dos resultados
    print("\nMatriz A:")
    print(np.array_str(A))
    print("\nVetor b:")
    print(b)
    print("\nSolução do sistema x:")
    print(x)

if __name__ == "__main__":
    main()
