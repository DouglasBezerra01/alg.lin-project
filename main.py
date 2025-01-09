import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import deque

# Função para gerar dados aleatórios
def gerar_dados(n):
    A = np.random.randint(1, 10, (n, n))  # Matriz A
    b = np.random.randint(1, 10, n)       # Vetor b
    return A, b

# Função para resolver o sistema linear usando decomposição LU
def resolver_sistema(A, b):
    P, L, U = scipy.linalg.lu(A)  # Decomposição LU
    y = scipy.linalg.solve(L, b)  # Resolução de Ly = b
    x = scipy.linalg.solve(U, y)  # Resolução de Ux = y
    return x

# Função para plotar os resultados
def plotar_resultados(A, b, x):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plotando a solução x como barras
    ax.bar(range(len(x)), x, label='Solução x', color='b')
    
    # Plotando o vetor b como linha
    ax.plot(range(len(A)), b, label='Resultado b', color='r', linestyle='--', marker='o')
    
    # Adicionando rótulos aos eixos
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor')
    
    # Adicionando título
    ax.set_title('Solução do Sistema Linear e Resultado b')
    
    # Adicionando legenda
    ax.legend()
    
    # Adicionando anotações para destacar pontos específicos
    for i, (xi, bi) in enumerate(zip(x, b)):
        ax.annotate(f'{xi:.2f}', (i, xi), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        ax.annotate(f'{bi:.2f}', (i, bi), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
    
    # Adicionando grade
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Mostrando a figura
    plt.show()

# Função para armazenar a solução em uma fila
def armazenar_em_fila(x):
    fila = deque(np.float64(x))
    return fila

# Função para validar a entrada do usuário
def validar_entrada():
    while True:
        try:
            n = int(input("Digite o tamanho da matriz (n): "))  # Tamanho da matriz
            if n > 0:
                return n
            else:
                print("Por favor, insira um número maior que zero.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número inteiro.")

# Função para permitir a entrada manual de A e b
def entrada_manual(n):
    A = np.zeros((n, n), dtype=int)
    b = np.zeros(n, dtype=int)
    print("Digite os elementos da matriz A:")
    for i in range(n):
        for j in range(n):
            A[i, j] = int(input(f"A[{i}][{j}] = "))
    print("Digite os elementos do vetor b:")
    for i in range(n):
        b[i] = int(input(f"b[{i}] = "))
    return A, b

def main():
    # Entradas
    n = validar_entrada()

    # Escolha entre gerar dados aleatórios ou entrada manual
    escolha = input("Deseja gerar dados aleatórios (A) ou inserir manualmente (M)? [Padrão: A] ").strip().upper() or 'A'
    if escolha == 'M':
        A, b = entrada_manual(n)
    else:
        A, b = gerar_dados(n)

    # Verificar se a matriz A é invertível
    if np.linalg.det(A) == 0:
        print("A matriz A não é invertível. Por favor, insira outra matriz.")
        return

    # Resolução do sistema
    x = resolver_sistema(A, b)

    # Armazenando solução em uma fila
    fila_solucoes = armazenar_em_fila(x)
    print("\nFila com a solução do sistema x:")
    print(f"deque([{', '.join(f'{v:.2f}' for v in fila_solucoes)}])")

    # Visualização
    plotar_resultados(A, b, x)

    # Exibição dos resultados
    print("\nMatriz A:")
    print(np.array_str(A))
    print("\nVetor b:")
    print(b)
    print("\nSolução do sistema x:")
    print(x)

if __name__ == "__main__":
    main()
