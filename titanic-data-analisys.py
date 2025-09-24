# Projeto: Análise Exploratória de Dados (Exemplo: Titanic)
# Autor: [Ash A. Orłowska]
# Descrição: Pequeno projeto de Data Science para GitHub.

import sys
import pathlib as Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    try: 
        df = sns.load_dataset("titanic")
        print("Dataset carregado do Seaborn.")
        return df
    except Exception as e:
        print("Falha ao carregar dataset do Seaborn:", e)
        local = Path("titanic.csv")
        if local.exists():
            print("Carregando 'titanic.csv' local.")
            return pd.read_csv(local)
        else:
            print("Nenhum dataset encontrado. Coloque 'titanic.csv' na pasta do projeto ou instale seaborn com datasets.")
            sys.exit(1)


def main():
    data = load_data()

    print("===== Primeiras Linhas =====")
    print(data.head().to_string())

    print("\n===== Informações do Dataset =====")
    data.info()

    print("\n===== Estatísticas Descritivas =====")
    print(data.describe(include="all").transpose())

    # Sobrevivência por gênero
    if 'sex' in data.columns and 'survived' in data.columns:
        survival_gender = data.groupby("sex")["survived"].mean()
        print("\n===== Taxa de Sobrevivência por Gênero =====")
        print(survival_gender)
    else:
        print("\nColunas 'sex' ou 'survived' não encontradas no dataset.")

    # Visualização: taxa de sobrevivência por classe (aceita 'class' ou 'pclass')
    class_col = None
    for c in ('class','pclass'):
        if c in data.columns:
            class_col = c
            break

    if class_col and 'survived' in data.columns:
        plt.figure(figsize=(6,4))
        sns.barplot(x=class_col, y="survived", data=data)
        plt.title("Taxa de Sobrevivência por Classe")
        plt.ylabel("Taxa Média de Sobrevivência")
        plt.xlabel(class_col.capitalize())
        plt.tight_layout()
        plt.savefig("survival_by_class.png")
        plt.show()
    else:
        print("\nNão foi possível plotar sobrevivência por classe (coluna 'class'/'pclass' ou 'survived' ausente).")

    # Visualização: distribuição de idade
    age_col = 'age' if 'age' in data.columns else 'Age' if 'Age' in data.columns else None
    if age_col:
        plt.figure(figsize=(6,4))
        sns.histplot(data[age_col].dropna(), bins=30, kde=True)
        plt.title("Distribuição das Idades")
        plt.xlabel("Idade")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig("age_distribution.png")
        plt.show()
    else:
        print("\nColuna de idade não encontrada para plotagem.")

    print("\nConclusão: Mulheres e passageiros de classes altas tiveram maiores chances de sobrevivência (visão geral).")


if __name__ == "__main__":
    main() 