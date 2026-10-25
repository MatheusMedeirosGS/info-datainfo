# Pasta de bases de conhecimento

Subpastas, uma por formato. O prompt  lê qualquer arquivo ou a pasta inteira:

    carregar C:/caminho/base_conhecimento           (pasta inteira, todas as subpastas)
    carregar C:/caminho/base_conhecimento/csv/ecologia_marinha.csv

No Colab, anexe base_conhecimento.zip pelo clipe: ele é aberto e lido como pasta.

| Subpasta | Arquivo | O que mostra |
|---|---|---|
| txt | base_animais.txt | base completa: declarações @, uma regra e triplas separadas por barra vertical |
| txt | frases_aves.txt | frases em português comum, uma por linha |
| csv | taxonomia_mamiferos.csv | cabeçalho, separador vírgula, ponto decimal |
| csv | ativos_exemplo_larga.csv | tabela larga fictícia de inventário (uma linha por ativo); o Info pergunta se leu certo e depois cobra patrimônio, setor e responsável em branco |
| csv | ecologia_marinha.csv | cabeçalho com acento, separador ; e decimal com vírgula |
| json | animais_nativo.json | formato que o próprio sistema grava (fatos, regras, ontologia) |
| json | repteis_lista.json | lista de objetos sujeito, relacao, objeto, confianca |
| json | insetos_por_entidade.json | um objeto por entidade, com as relações dentro |
| json | anfibios_arrays.json | lista de listas [sujeito, relação, objeto, confiança] |
| pdf | relatorio_fauna.pdf | texto corrido, frases simples |
| pdf | tabela_triplas.pdf | linhas de tabela separadas por barra vertical |

De propósito, ecologia_marinha.csv diz que Golfinho vive em Rio, e base_animais.txt diz Oceano:
ao ler a pasta inteira a IA percebe a contradição e pergunta qual vale.
