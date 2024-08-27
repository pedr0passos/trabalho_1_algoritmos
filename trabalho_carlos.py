import sys
import time

def ler_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r') as arquivo:
        try:
            # Lê a primeira linha e distribui os valores
            primeira_linha = arquivo.readline().strip().split()
            quantidade_sistemas = int(primeira_linha[0])
            dimensao = int(primeira_linha[1])
            precisao = float(primeira_linha[2])

            # Lê as próximas linhas e forma a matriz A
            matriz_A = []
            for _ in range(dimensao):
                linha = list(map(float, arquivo.readline().strip().split()))
                matriz_A.append(linha)

            # Lê a última linha, que é o vetor B
            vetores_B = []
            for _ in range(quantidade_sistemas):
                vetor_B = list(map(float, arquivo.readline().strip().split()))
                vetores_B.append(vetor_B)
                
            

            return quantidade_sistemas, dimensao, precisao, matriz_A, vetores_B
        except:
            print("Erro ao ler arquivo")

def Eliminacao_Gauss(dimensao, matriz_A, vetores_B):

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

def Fatoracao_LU_resolver(dimensao, matriz_A, vetores_B):
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

        print("Metodo Eliminação de Gauss")
        solucoes, tempos = Eliminacao_Gauss(dimensao, matriz_A, vetores_B)
        print(f"Soluções = {solucoes}")
        print(f"tempos = {tempos}")

        print("Método Fatoração LU")
        solucoes, tempos = Fatoracao_LU_resolver(dimensao, matriz_A, vetores_B)
        print(f"Soluções = {solucoes}")
        print(f"tempos = {tempos}")

    except Exception as error:
        print(error)



main()