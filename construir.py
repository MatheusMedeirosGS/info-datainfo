"""Gera os artefatos a partir de ia_conhecimento.py (a fonte única).

    python construir.py

Produz:
  notebooks/ia_conhecimento.ipynb  notebook para o Google Colab (uma célula por bloco "# %% [n] título")
  info_datainfo.py       módulo importável (só as células de biblioteca, sem demos nem exemplo)

As células são renumeradas em sequência a cada execução, então basta inserir uma célula nova
com qualquer número na fonte.
"""
import json
import os
import re

PASTA = os.path.dirname(os.path.abspath(__file__))
FONTE = os.path.join(PASTA, "ia_conhecimento.py")
CABECALHO = re.compile(r"^# %% \[\d+\] (.*)$", re.M)

# células que ficam de fora do módulo: demonstrações da aula, arquivo de exemplo e a abertura do chat
FORA_DO_MODULO = ("DEMO", "Arquivo de exemplo", "Abrir o prompt")


def ler_celulas(texto):
    partes = CABECALHO.split(texto)
    return partes[0], list(zip(partes[1::2], partes[2::2]))


def renumerar(texto):
    prefixo, celulas = ler_celulas(texto)
    return prefixo + "".join(f"# %% [{i}] {titulo}{corpo}" for i, (titulo, corpo) in enumerate(celulas, 1))


def notebook(celulas):
    md = lambda linhas: {"cell_type": "markdown", "metadata": {}, "source": linhas}
    blocos = [md(["# Info, da DataInfo\n",
                  "Execute as células em ordem (Ambiente de execução > Executar tudo). O chat abre na "
                  "última célula."])]
    for titulo, corpo in celulas:
        blocos.append(md([f"## {titulo}"]))
        blocos.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": corpo.strip("\n").splitlines(keepends=True)})
    return {"cells": blocos, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def modulo(celulas):
    corpo = "\n\n\n".join(f"# --- {titulo} ---{c.rstrip()}" for titulo, c in celulas
                          if not titulo.startswith(FORA_DO_MODULO)
                          and not re.match(r"^(DEMO)", titulo))
    return ('"""Info, da DataInfo: base de conhecimento que conversa, pergunta e aprende.\n\n'
            'Gerado por construir.py a partir de ia_conhecimento.py. Não edite este arquivo.\n'
            'Uso:  python -m info_datainfo [arquivo]\n"""\n\n' + corpo + '''


def main(argumentos=None):
    import sys
    argumentos = sys.argv[1:] if argumentos is None else argumentos
    prompt_terminal(argumentos[0] if argumentos else None)


if __name__ == "__main__":
    main()
''')


def main():
    with open(FONTE, encoding="utf-8") as fp:
        texto = renumerar(fp.read())
    provisorio = FONTE + ".tmp"   # grava inteiro antes de trocar, para não perder a fonte única
    with open(provisorio, "w", encoding="utf-8") as fp:
        fp.write(texto)
    os.replace(provisorio, FONTE)
    _, celulas = ler_celulas(texto)
    with open(os.path.join(PASTA, "notebooks", "ia_conhecimento.ipynb"), "w", encoding="utf-8") as fp:
        json.dump(notebook(celulas), fp, ensure_ascii=False, indent=1)
    bib = [(t, c) for t, c in celulas if not t.startswith(FORA_DO_MODULO)]
    # as demonstrações (células 7 a 17) usam o mesmo estado global e não fazem parte da biblioteca
    bib = [(t, c) for t, c in bib if not re.match(r"^DEMO", t)]
    with open(os.path.join(PASTA, "info_datainfo.py"), "w", encoding="utf-8") as fp:
        fp.write(modulo(bib))
    print(f"{len(celulas)} células; módulo com {len(bib)}.")


if __name__ == "__main__":
    main()
