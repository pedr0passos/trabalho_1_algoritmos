import numpy as np
import time
import sys
from pprint import pprint

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

            return dimensao, precisao, matriz_a, vetores_b
        except:
            print("Erro ao ler arquivo")

def metodo_gauss_jacobi(dimensao, precisao, matriz_A, vetores_B):

def gauss_seidel(matriz_A, vetores_B, precisao):


def eliminacao_gauss(dimensao, matriz_A, vetores_B):

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

def fatoracao_lu_resolver(matriz_A, vetores_B):
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
                if U[i][i] == 0:
                    raise ValueError("Divisão por zero detectada durante a fatoração LU.")
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

def formatar_lista(lista):
    """Formata uma lista de números para 10 casas decimais."""
    return [round(elem, 10) if isinstance(elem, (int, float)) else elem for elem in lista]

def formatar_matriz(matriz):
    """Formata uma matriz para exibir os valores com 10 casas decimais."""
    return [formatar_lista(linha) for linha in matriz]

def imprimir_resultados(idx, A, b, precisao, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel):

    print('------------------------------------------------------')
    print(f"Vetor B: {idx + 1}")
    print('------------------------------------------------------')
    print("\nEliminação de Gauss:")
    print("Solução:")
    pprint(formatar_lista(X_gauss[idx]))
    print(f"Tempo de execução: {tempo_gauss[idx]:.10f} segundos")
    print('------------------------------------------------------')
    print("\nFatoração LU:")
    print("Solução:")
    pprint(formatar_lista(X_lu[idx]))
    print(f"Tempo de execução: {tempo_lu[idx]:.10f} segundos")
    print('------------------------------------------------------')
    print("\nMétodo de Gauss-Jacobi:")
    print("Solução:")
    pprint(formatar_lista(X_jacobi[idx]))
    print(f"Tempo de execução: {tempo_jacobi[idx]:.10f} segundos")
    print('------------------------------------------------------')
    print("\nMétodo de Gauss-Seidel:")
    print("Solução:")
    pprint(formatar_lista(X_seidel[idx]))
    print(f"Tempo de execução: {tempo_seidel[idx]:.10f} segundos")

def main():
    try:
        nome_arquivo = "2_3x3.txt" #recupera o nome do arquivo no argumento

        dimensao, precisao, matriz_A, vetores_B = ler_arquivo(nome_arquivo)

        X_gauss, tempo_gauss = eliminacao_gauss(dimensao, matriz_A, vetores_B)
        X_lu, tempo_lu = fatoracao_lu_resolver(matriz_A, vetores_B)
        
        X_jacobi, tempo_jacobi = metodo_gauss_jacobi(dimensao, precisao, matriz_A, vetores_B)
        X_seidel, tempo_seidel = gauss_seidel(matriz_A, vetores_B, precisao)

        for idx in range(len(vetores_B)):
            imprimir_resultados(idx, matriz_A, vetores_B[idx], precisao, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel)
        print('------------------------------------------------------')
    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()