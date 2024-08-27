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

def gauss_jacobi(matriz_A, vetores_B, iteracoes):
    dimensao = len(matriz_A)
    solucoes = []
    tempos = []
    
    for vetor_B in vetores_B:
        tempo_inicial = time.time()
        iteracao = 0
        auxiliar = np.zeros(dimensao)
        solucao = np.zeros(dimensao)
        
        while iteracao < iteracoes:
            for linha in range(dimensao):
                x = vetor_B[linha]
                for coluna in range(dimensao):
                    if linha != coluna:
                        x -= (matriz_A[linha][coluna] * solucao[coluna])
                x /= matriz_A[linha][linha]
                auxiliar[linha] = x
            iteracao += 1

            for i in range(len(auxiliar)):
                solucao[i] = auxiliar[i]

        tempo_final = time.time()
        tempo = tempo_final - tempo_inicial

        tempos.append(tempo)
        solucoes.append(solucao)
    
    return solucoes, tempos


def gauss_seidel(matriz_A, vetores_B, iteracoes=1000):
    dimensao = len(matriz_A)
    solucoes = []
    for vetor_B in vetores_B:
        solucao = np.zeros(dimensao)
        iteracao = 0
        while iteracao < iteracoes:
            for linha in range(dimensao):
                x = vetor_B[linha]
                for coluna in range(dimensao):
                    if linha != coluna:
                        x -= (matriz_A[linha][coluna] * solucao[coluna])
                x /= matriz_A[linha][linha]
                solucao[linha] = x
            iteracao += 1
        solucoes.append(solucao)
    return solucoes, iteracoes
    

def eliminacao_Gauss(dimensao, matriz_A, vetores_B):

    solucoes = []
    tempos_execucao = []

    for vetor_B in vetores_B:
        inicio_tempo = time.time()

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

        tempo_execucao = time.time() - inicio_tempo

        solucoes.append(x)
        tempos_execucao.append(tempo_execucao)

    return solucoes, tempos_execucao

def fatoracao_LU_resolver(dimensao, matriz_A, vetores_B):
    solucoes = []
    tempos_execucao = []

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
        inicio_tempo = time.time()

        solucao = resolver_sistema_lu(L, U, vetor_B)
        tempo_execucao = time.time() - inicio_tempo

        solucoes.append(solucao)
        tempos_execucao.append(tempo_execucao)

    return solucoes, tempos_execucao

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

def imprime_resultados_separados(metodo, solucao, tempo):
    print(metodo + ": ")
    print("Solução: ")
    print(solucao)
    print("Tempo: ")
    print(tempo)
    print("------------------------------------------------------------")
    


def main():
    quantidade_sistemas, dimensao, precisao, matriz_a, vetores_b = ler_arquivo('2_3x3.txt')
    print("Quantidade de Sistemas:",quantidade_sistemas)
    print("Dimensão:",dimensao)
    print("Precisão:",precisao)
    print("Matriz A:",matriz_a)
    print("Vetor B:",vetores_b)
    print("------------------------------------------------------------")

    X_gauss, tempo_gauss = eliminacao_Gauss(dimensao, matriz_a, vetores_b)
    imprime_resultados_separados("Eliminação de gauss", X_gauss, tempo_gauss)


    X_lu, tempo_lu = fatoracao_LU_resolver(dimensao, matriz_a, vetores_b)
    imprime_resultados_separados("Fatoração LU", X_lu, tempo_lu)


    X_jacodi, tempo_jacodi = gauss_jacobi(matriz_a, vetores_b, 100)
    imprime_resultados_separados("Gauss Jacodi", X_jacodi, tempo_jacodi)
    
    X_seidel, tempo_seidel = gauss_seidel(matriz_a, vetores_b, 1000)
    imprime_resultados_separados("Gauss Seidel", X_seidel, tempo_seidel)   
    


if __name__ == '__main__':
    main()