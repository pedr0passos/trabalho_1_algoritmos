import numpy as np
import time

"""
@authors: 
Pedro Henrique Passos Rocha,
Catterina Salvador, 
Carlos Victor
"""

def corrigir_caracteres(texto):
    return texto.replace('−', '-')

def ler_arquivo(nome_arquivo):
    
    with open(nome_arquivo, 'r', encoding='utf-8') as file:
        try:
            linha = file.readline().split()
            quantidade_sistemas = int(linha[0])
            dimensao = int(corrigir_caracteres(linha[1]))
            precisao = float(corrigir_caracteres(linha[2]))

            matriz_a = []
            for _ in range(dimensao):
                linha = list(map(float, corrigir_caracteres(file.readline()).split()))
                matriz_a.append(linha)                
            
            vetores_b = []
            for _ in range(quantidade_sistemas):
                vetor_b = list(map(float, corrigir_caracteres(file.readline()).split()))
                vetores_b.append(vetor_b)

            return quantidade_sistemas, dimensao, precisao, matriz_a, vetores_b
        except:
            print("Erro ao ler arquivo")

def gauss_jacobi(matriz_A, vetor_B, precisao, max_iteracoes=1000):
    dimensao = len(vetor_B)
    x = np.zeros(dimensao)
    x_anterior = np.zeros(dimensao)
    erro = np.inf
    iteracao = 0

    start_time = time.time()

    while erro > precisao and iteracao < max_iteracoes:
        for i in range(dimensao):
            soma = 0
            for j in range(dimensao):
                if j != i:
                    soma += matriz_A[i][j] * x[j]
            x[i] = (vetor_B[i] - soma) / matriz_A[i][i]
        
        erro = np.linalg.norm(x - x_anterior)
        x_anterior = np.copy(x)
        iteracao += 1

    end_time = time.time()  
    tempo = end_time - start_time  

    return x, tempo

def gauss_seidel(matriz_A, vetor_B, precisao, max_iteracoes=1000):
    dimensao = len(vetor_B)
    x = np.zeros(dimensao)
    x_anterior = np.zeros(dimensao)
    erro = np.inf
    iteracao = 0

    start_time = time.time()

    while erro > precisao and iteracao < max_iteracoes:
        for i in range(dimensao):
            soma = 0
            for j in range(dimensao):
                if j != i:
                    soma += matriz_A[i][j] * x[j]
            x[i] = (vetor_B[i] - soma) / matriz_A[i][i]
        
        erro = np.linalg.norm(x - x_anterior)
        x_anterior = np.copy(x)
        iteracao += 1

    end_time = time.time()  
    tempo = end_time - start_time  

    return x, tempo

def metodo_gauss(matriz_A, vetor_B):
    dimensao = len(vetor_B)
    start_time = time.time()

    for i in range(dimensao):
        pivo = matriz_A[i][i]
        for j in range(i + 1, dimensao):
            multiplicador = matriz_A[j][i] / pivo
            matriz_A[j] -= multiplicador * matriz_A[i]
            vetor_B[j] -= multiplicador * vetor_B[i]
    
    x = np.zeros(dimensao)
    for i in range(dimensao - 1, -1, -1):
        x[i] = vetor_B[i]
        for j in range(i + 1, dimensao):
            x[i] -= matriz_A[i][j] * x[j]
        x[i] /= matriz_A[i][i]
    
    end_time = time.time()
    tempo = end_time - start_time

    return x, tempo

def metodo_lu(matriz_A, vetor_B):
    dimensao = len(vetor_B)
    start_time = time.time()

    L = np.eye(dimensao)
    U = np.copy(matriz_A)

    for i in range(dimensao):
        pivo = U[i][i]
        for j in range(i + 1, dimensao):
            multiplicador = U[j][i] / pivo
            L[j][i] = multiplicador
            U[j] -= multiplicador * U[i]
    
    y = np.zeros(dimensao)
    for i in range(dimensao):
        y[i] = vetor_B[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]
    
    x = np.zeros(dimensao)
    for i in range(dimensao - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, dimensao):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    
    end_time = time.time()
    tempo = end_time - start_time

    return x, tempo

def imprimir_resultados(idx, A, b, precisao, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel):
    print(f"Sistema {idx + 1}:")
    print("Matriz A:")
    print(A)
    print("Vetor B:")
    print(b)
    print(f"Precisão: {precisao:.3f}")
    
    print("\nMétodo de Eliminação de Gauss:")
    print(f"Vetor X:\n{X_gauss}")
    print(f"Tempo: {tempo_gauss:.4f} segundos")
    
    print("\nMétodo de Fatoração LU:")
    print(f"Vetor X:\n{X_lu}")
    print(f"Tempo: {tempo_lu:.4f} segundos")
    
    print("\nMétodo de Gauss-Jacobi:")
    print(f"Vetor X:\n{X_jacobi}")
    print(f"Tempo: {tempo_jacobi:.4f} segundos")
    
    print("\nMétodo de Gauss-Seidel:")
    print(f"Vetor X:\n{X_seidel}")
    print(f"Tempo: {tempo_seidel:.4f} segundos")
    
    print()

def main():
    quantidade_sistemas, dimensao, precisao, matriz_a, vetores_b = ler_arquivo('2_3x3.txt')
    print("Quantidade de Sistemas:",quantidade_sistemas)
    print("Dimensão:",dimensao)
    print("Precisão:",precisao)
    print("Matriz A:",matriz_a)
    print("Vetor B:",vetores_b)

if __name__ == '__main__':
    main()