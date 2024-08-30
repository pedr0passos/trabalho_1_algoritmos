import numpy as np
import time
import sys
import math
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

def gauss_jacobi(matriz_A,vetores_b,precisao, maximo_iteracoes=1000):
    solucoes = []
    for b in vetores_b:
        n = len(b)
        convergiu = True
        x = b.copy()

        x = [b[i] / matriz_A[i][i] if matriz_A[i][i] != 0 else 0 for i in range(n)]
        
        if not all(matriz_A[i][i] != 0 for i in range(n)):
            convergiu = False
        
        if convergiu:
            xk = x.copy()
            iteracao    = 0
        
            while (iteracao < maximo_iteracoes):
                iteracao = iteracao + 1
                for i in list(range(1,n+1,1)):
                    s = 0
                    for j in list(range(1,n+1,1)):
                        if ((i-1) != (j-1)):
                            s = s + matriz_A[i-1][j-1]*x[j-1]

                    xk[i-1] = (1/matriz_A[i-1][i-1])*(b[i-1]-s)
                
                if comparar(x,xk,precisao):
                    x = xk.copy()
                    break    
                x = xk.copy()

        solucoes.append(x)
        
    return solucoes

def comparar(x,xk,eps):
  soma = 0
  zip_object = zip(x, xk)
  for list1_i, list2_i in zip_object:
    soma = soma + math.fabs(list1_i-list2_i)

  if (soma < eps):
    return True
  else:
    return False  
  
def gauss_seidel(matriz_A, vetores_b, precisao, maximo_iteracoes=1000):
    n = len(matriz_A)
    solucoes = []
    
    for b in vetores_b:
        convergiu = True
        x = [0.0] * n
        
        for i in range(n):
            if math.fabs(matriz_A[i][i]) > 0.0:
                x[i] = b[i] / matriz_A[i][i]
            else:
                convergiu = False
                break
        
        if convergiu:
            iteracao = 0
            xk = x.copy()
            
            while iteracao < maximo_iteracoes:
                iteracao += 1
                for i in range(n):
                    s = 0
                    for j in range(n):
                        if i > j:
                            s += matriz_A[i][j] * xk[j]
                        elif i < j:
                            s += matriz_A[i][j] * x[j]
                    
                    xk[i] = (1 / matriz_A[i][i]) * (b[i] - s)
                
                if comparar(x, xk, precisao):
                    x = xk.copy()
                    break
                
                x = xk.copy()
            
            solucoes.append(x)
        else:
            solucoes.append(None)
    
    return solucoes
                
def eliminacao_gauss(dimensao, matriz_A, vetores_B):
    solucoes = []

    for vetor_B in vetores_B:
        matriz_ab = [matriz_A[i] + [vetor_B[i]] for i in range(dimensao)]
        
        for i in range(dimensao):
            max_el = abs(matriz_ab[i][i])
            max_linha = i
            for k in range(i + 1, dimensao):
                if abs(matriz_ab[k][i]) > max_el:
                    max_el = abs(matriz_ab[k][i])
                    max_linha = k

            matriz_ab[i], matriz_ab[max_linha] = matriz_ab[max_linha], matriz_ab[i]

            for k in range(i + 1, dimensao):
                c = -matriz_ab[k][i] / matriz_ab[i][i]
                for j in range(i, dimensao + 1):
                    if i == j:
                        matriz_ab[k][j] = 0
                    else:
                        matriz_ab[k][j] += c * matriz_ab[i][j]
        
        x = [0 for _ in range(dimensao)]
        for i in range(dimensao - 1, -1, -1):
            x[i] = matriz_ab[i][dimensao] / matriz_ab[i][i]
            for k in range(i - 1, -1, -1):
                matriz_ab[k][dimensao] -= matriz_ab[k][i] * x[i]

        solucoes.append(x)

    return solucoes

def fatoracao_lu_resolver(matriz_A, vetores_B):
    solucoes = []

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

        for i in range(n):
            y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

        return x

    L, U = fatoracao_lu(matriz_A)

    for vetor_B in vetores_B:

        solucao = resolver_sistema_lu(L, U, vetor_B)
        solucoes.append(solucao)

    return solucoes

def formatar_lista(lista):
    return [round(elem, 10) if isinstance(elem, (int, float)) else elem for elem in lista]

def formatar_matriz(matriz):
    return [formatar_lista(linha) for linha in matriz]

def imprimir_resultados(idx, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel):

    print('------------------------------------------------------')
    print(f"Vetor B: {idx + 1}")
    print('------------------------------------------------------')
    print("\nEliminação de Gauss:")
    print("Solução:")
    pprint(formatar_lista(X_gauss[idx]))
    print(f"Tempo de execução: {tempo_gauss:.10f} segundos")
    print('------------------------------------------------------')
    print("\nFatoração LU:")
    print("Solução:")
    pprint(formatar_lista(X_lu[idx]))
    print(f"Tempo de execução: {tempo_lu:.10f} segundos")
    print('------------------------------------------------------')
    print("\nMétodo de Gauss-Jacobi:")
    print("Solução:")
    pprint(formatar_lista(X_jacobi[idx]))
    print(f"Tempo de execução: {tempo_jacobi:.10f} segundos")
    print('------------------------------------------------------')
    print("\nMétodo de Gauss-Seidel:")
    print("Solução:")
    pprint(formatar_lista(X_seidel[idx]))
    print(f"Tempo de execução: {tempo_seidel:.10f} segundos")

def main():
    try:
        nome_arquivo = sys.argv[1]

        dimensao, precisao, matriz_A, vetores_B = ler_arquivo(nome_arquivo)
        
        start = time.time()
        X_gauss = eliminacao_gauss(dimensao, matriz_A, vetores_B)
        tempo_gauss = time.time() - start 

        start = time.time()
        X_lu = fatoracao_lu_resolver(matriz_A, vetores_B)
        tempo_lu = time.time() - start

        start = time.time()
        X_jacobi  = gauss_jacobi(matriz_A, vetores_B, precisao)
        tempo_jacobi = time.time() - start

        start = time.time()
        X_seidel  = gauss_seidel(matriz_A, vetores_B, precisao)
        tempo_seidel = time.time() - start

        for idx in range(len(vetores_B)):
            imprimir_resultados(idx, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel)
        print('------------------------------------------------------')
    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()