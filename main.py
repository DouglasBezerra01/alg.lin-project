import numpy as np # type: ignore
import scipy.linalg # type: ignore
import matplotlib.pyplot as plt # type: ignore

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
    
    ax.bar(range(len(x)), x, label='Solução x', color='b')
    ax.plot(range(len(A)), b, label='Resultado b', color='r', linestyle='--')
    
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor')
    ax.set_title('Solução do Sistema Linear e Resultado b')
    ax.legend()
    plt.show()

# Entradas
n = 3  # Valor padrão para o tamanho da matriz
# try:
#     n = int(input("Digite o tamanho da matriz (n): "))  # Tamanho da matriz
# except ValueError:
#     print("Entrada inválida. Usando valor padrão n = 3.")
#     n = 3

A, b = gerar_dados(n)

# Resolução do sistema
x = resolver_sistema(A, b)

# Visualização
plotar_resultados(A, b, x)

# Exibição dos resultados
print("Matriz A:")
print(A)
print("\nVetor b:")
print(b)
print("\nSolução do sistema x:")
print(x)
