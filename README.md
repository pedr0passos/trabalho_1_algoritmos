# Trabalho de Sistemas Lineares

Este repositório contém um código Python para a resolução de sistemas lineares utilizando quatro métodos diferentes. O código lê os dados de um arquivo e aplica os métodos de resolução aos sistemas especificados. A seguir estão listadas as funções presentes no código:

- **corrigir_caracteres(texto)**: Substitui caracteres especiais no texto.
- **ler_arquivo(nome_arquivo)**: Lê dados de um arquivo e extrai informações sobre os sistemas lineares.
- **gauss_jacobi(matriz_A, vetor_B, precisao, max_iteracoes=1000)**: Resolve o sistema linear usando o Método de Gauss-Jacobi.
- **gauss_seidel(matriz_A, vetor_B, precisao, max_iteracoes=1000)**: Resolve o sistema linear usando o Método de Gauss-Seidel.
- **metodo_gauss(matriz_A, vetor_B)**: Resolve o sistema linear usando o Método de Eliminação de Gauss.
- **metodo_lu(matriz_A, vetor_B)**: Resolve o sistema linear usando o Método de Fatoração LU.
- **imprimir_resultados(idx, A, b, precisao, X_gauss, tempo_gauss, X_lu, tempo_lu, X_jacobi, tempo_jacobi, X_seidel, tempo_seidel)**: Imprime os resultados obtidos pelos diferentes métodos.
- **main()**: Função principal que coordena a execução do código, lendo o arquivo e executando os métodos de resolução.

## Uso

1. Prepare um arquivo de entrada no formato especificado.
2. Execute o script Python para ler o arquivo e resolver os sistemas lineares.
3. Os resultados dos métodos serão exibidos no console.

Certifique-se de ter o NumPy instalado para executar o código.
