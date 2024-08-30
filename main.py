import numpy as np
import time
import sys

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

def Eliminacao_Gauss(dimensao, matriz_A, vetores_B):

    solucoes = []

    start_time = time.time()

    for vetor_B in vetores_B:

        # Copia a matriz_A para não modificar o original
        matriz_ab = [matriz_A[i] + [vetor_B[i]] for i in range(dimensao)]

        # Eliminação de Gauss
        for i in range(dimensao):
            max_el = abs(matriz_ab[i][i])
            max_row = i
            for k in range(i + 1, dimensao):
                if abs(matriz_ab[k][i]) > max_el:
                    max_el = abs(matriz_ab[k][i])
                    max_row = k

            matriz_ab[i], matriz_ab[max_row] = matriz_ab[max_row], matriz_ab[i]

            for k in range(i + 1, dimensao):
                c = -matriz_ab[k][i] / matriz_ab[i][i]
                for j in range(i, dimensao + 1):
                    if i == j:
                        matriz_ab[k][j] = 0
                    else:
                        matriz_ab[k][j] += c * matriz_ab[i][j]

        # Substituição Regressiva
        x = [0 for _ in range(dimensao)]
        for i in range(dimensao - 1, -1, -1):
            x[i] = matriz_ab[i][dimensao] / matriz_ab[i][i]
            for k in range(i - 1, -1, -1):
                matriz_ab[k][dimensao] -= matriz_ab[k][i] * x[i]

        end_time = time.time()
        tempo_execucao = end_time - start_time  
        solucoes.append(x)

    return solucoes, tempo_execucao

def Fatoracao_LU_resolver(dimensao, matriz_A, vetores_B):
    solucoes = []
    start_time = time.time()

    def fatoracao_lu(matriz_A):

        n = len(matriz_A)
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            L[i][i] = 1.0
            for j in range(i, n):
                soma = sum(U[k][j] * L[i][k] for k in range(i))
                U[i][j] = matriz_A[i][j] - soma
            for j in range(i + 1, n):
                soma = sum(U[k][i] * L[j][k] for k in range(i))
                L[j][i] = (matriz_A[j][i] - soma) / U[i][i]

        return L, U

    def resolver_sistema_lu(L, U, b):

        n = len(L)
        y = [0.0] * n
        x = [0.0] * n

        # Solução de Ly = b (substituição para frente)
        for i in range(n):
            y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

        # Solução de Ux = y (substituição para trás)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

        return x

    # Fatoração LU da matriz A
    L, U = fatoracao_lu(matriz_A)

    # Resolver para cada vetor B
    for vetor_B in vetores_B:

        solucao = resolver_sistema_lu(L, U, vetor_B)
        solucoes.append(solucao)
    
    end_time = time.time()
    tempo_execucao = end_time - start_time  

    return solucoes, tempo_execucao

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

    print("TRABALHO DE ALGORITMOS")

    try:
        nome_arquivo = sys.argv[1] #recupera o nome do arquivo no argumento

        quantidade_sistemas, dimensao, precisao, matriz_A, vetores_B = ler_arquivo(nome_arquivo)

        print(f"quantidade de sistemas = {quantidade_sistemas}")
        print(f"dimensao = {dimensao}")
        print(f"precisao = {precisao}")
        print(f"matrizA = {matriz_A}")
        print(f"Vetor B = {vetores_B}")

        print(f"tempo agora = {time.time()}")

        start_time = time.time()
        print("Metodo Eliminação de Gauss")
        solucoes, tempo = Eliminacao_Gauss(dimensao, matriz_A, vetores_B)
        end_time = time.time()
        print(f"Soluções = {solucoes}")
        print(f"tempos = {(end_time - start_time)}")

        start_time = time.time()
        print("Método Fatoração LU")
        solucoes, tempo = Fatoracao_LU_resolver(dimensao, matriz_A, vetores_B)
        end_time = time.time()
        print(f"Soluções = {solucoes}")
        print(f"tempos = {(end_time - start_time)}")

        print("Gauss Jacobi")
        solucoes, tempo = gauss_jacobi(matriz_A, vetores_B, precisao)
        print(f"Soluções = {solucoes}")
        print(f"tempos = {tempo}")

        print("Gauss seidel")
        solucoes, tempo = gauss_seidel(matriz_A, vetores_B, precisao)
        print(f"Soluções = {solucoes}")
        print(f"tempos = {tempo}")

    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()