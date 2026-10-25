# Info, da DataInfo

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatheusMedeirosGS/info-datainfo/blob/main/notebooks/ia_conhecimento.ipynb)

Base de conhecimento que conversa: você carrega um arquivo (txt, csv, json, pdf, zip ou planilha de inventário), e o Info lê, deduz, pergunta o que falta, confronta contradições e grava o que aprende. Interface só de texto, sem menus.

Projeto acadêmico da disciplina de Inteligência Artificial (3º período de ADS). O nome e as cores da DataInfo vêm do portal público da empresa (datainfo.inf.br); o projeto não é oficial nem afiliado a ela, e o logotipo oficial não é usado.

## Uso

**Google Colab.** Abra `notebooks/ia_conhecimento.ipynb` e execute todas as células. O chat abre na última.

**Terminal.**

```bash
pip install numpy networkx matplotlib
python -m info_datainfo exemplos/base_animais.txt
```

**Como módulo.**

```python
import info_datainfo as info
d = info.Dialogo()
print(d.responder("carregar exemplos/base_animais.txt"))
```

Dentro do chat, `rode os testes` executa as verificações internas.

## O que ele faz

| Área | Exemplos de comandos |
|---|---|
| Aprender e perguntar | `carregar exemplos/base_animais.txt`, `o gato caça o rato`, `sugira`, `aprenda padrões` |
| Conversar | `fale sobre o gato`, `e ele vive onde?`, `o que o gato e o cachorro têm em comum?`, `quem se parece com o gato?` |
| Explicar | `por que o leão é um animal?`, `por que o pinguim não é um mamífero?`, `mostre seu raciocínio` |
| Imaginar | `e se a águia nada?`, `e se a regra heranca_de_classe não existisse?` (rodam numa cópia, não gravam) |
| Autoavaliar | `o que você não sabe?`, `avalie-se` (TransE e RotatE), `curva de calibração`, `métricas`, `rode os testes` |
| Arquivos | `carregar` e `exportar` em txt, md (tabelas `sujeito | relação | objeto` e perguntas `**P-001 Pergunta?** Resposta`), csv, json, pdf, graphml, zip, pasta e link |
| Explorar bases | `compare com bases/animais.txt`, `nomes parecidos`, `o que depende da regra X`, `perguntas frequentes`, `grafo interativo` (gera `grafo_info.html`) |
| Apresentar | `demonstração` (roteiro passo a passo, com `continuar`), `verificar ambiente`, `ata em word`, `fonte maior`, `alto contraste` |

Respostas como `provavelmente sim` ou `acho que não` valem menos que `sim` e `não`. O Info nunca afirma o que não tem como justificar: sem fato nem regra, ele diz que não sabe e pergunta.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `ia_conhecimento.py` | Fonte única, dividida em células `# %% [n] título` |
| `construir.py` | Gera o notebook e o módulo `info_datainfo.py` a partir da fonte |
| `info_datainfo.py` | Módulo gerado (não editar) |
| `notebooks/` | Notebook para o Colab (gerado) |
| `bases/` | 5 bases prontas (animais, redes, IA e computação, geografia do Brasil e de Pernambuco, sistema solar); veja `bases/LEIAME.md` |
| `exemplos/` | Bases de exemplo em txt, csv, json e pdf |
| `tests/` | Testes automáticos |

Para alterar o código, edite `ia_conhecimento.py` e rode `python construir.py`.

## Bases prontas

```text
carregar bases/animais.txt
carregar bases/sistema_solar.txt
```

Os fatos foram escritos a partir de conhecimento geral e fontes citadas em `bases/LEIAME.md`; confira antes de citar formalmente. Entidades com prefixo `Exemplo` são fictícias.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Dependências

`numpy`, `networkx`, `matplotlib`. Opcionais: `pymupdf` (PDF), `ipywidgets` (chat no notebook), `anthropic` (modo avançado, desligado por padrão e com consentimento explícito).

## Licença

MIT. Veja o arquivo `LICENSE`.
