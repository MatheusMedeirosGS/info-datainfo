# Bases de conhecimento do Info

Bases prontas para o Info, da DataInfo, carregar. Todas usam o mesmo formato de texto de
`exemplos/base_animais.txt` e são lidas pelo carregador atual (`importar_arquivo` e
`importar_pasta`) sem nenhuma alteração no código.

## Formato

```
# comentário
@transitiva fica_em                  propriedades das relações
@simetrica faz_fronteira_com
@inversa capital_de tem_capital
@funcional capital_de
@oposta tem_propriedade não_tem_propriedade
@disjuntas Estado Municipio
@regra nome: (?a rel ?b) & (?b rel2 ?c) => (?a rel3 ?c) 0.9
sujeito | relação | objeto | confiança (opcional, padrão 1.0)
```

Nomes de entidades estão em CamelCase sem acento (`RioDeJaneiro`, `PlanetaAnao`) para que o
leitor não junte nomes parecidos. Generalizações com exceção conhecida têm confiança menor
que 1 (por exemplo `Ave | tem_propriedade | voa | 0.8`) e a exceção aparece como fato direto
(`Avestruz | não_tem_propriedade | voa`).

## Bases

| Arquivo | Domínio | Fatos | Regras | Deduções após pensar |
|---|---|---:|---:|---:|
| `animais.txt` | taxonomia, características, dieta, predação, exceções | 171 | 6 | 789 |
| `redes_infraestrutura.txt` | modelo OSI, protocolos, portas, endereçamento, equipamentos | 117 | 5 | 79 |
| `ia_computacao.txt` | áreas da IA, algoritmos, marcos históricos, linguagens, complexidade | 129 | 4 | 51 |
| `geografia_brasil_pe.txt` | regiões, 27 UFs e capitais, fronteiras, Pernambuco | 184 | 4 | 299 |
| `sistema_solar.txt` | Sol, planetas, planetas anões, satélites | 89 | 6 | 187 |
| **Total (pasta inteira)** | | **690** | **25** | **1405** |

Cuidado ao escrever bases novas: não use `tem` nem `é` sozinhos como relação (colidem com
`tem_propriedade` e `é_um` no interpretador) e evite "nao" dentro do nome de uma propriedade
(uma propriedade `sla_nao_estourado` era lida como negação de `sla_estourado`).

"Deduções" conta os fatos inferidos com confiança de pelo menos 0,5 depois de `pensar()`.

### animais.txt
Hierarquia SerVivo > Animal > Vertebrado/Invertebrado > classes (Mamifero, Ave, Reptil,
Anfibio, Peixe, Inseto, Aracnideo, Crustaceo, Molusco) e subgrupos (Felino, Roedor, Cetaceo,
Quiroptero, Monotremado, Ratita, Pinguim, Serpente). Exceções reais: morcego voa,
ornitorrinco e equidna põem ovos, pinguim, avestruz, ema e kiwi não voam, peixe-boi é
mamífero. Regras: penas e ovos indicam ave, amamentar indica mamífero, predadores da mesma
presa competem, ave aquática, herança de não_tem_propriedade.
Perguntas para testar: "o morcego voa?", "o avestruz voa?", "a capivara tem coluna vertebral?".

### redes_infraestrutura.txt
Sete camadas do OSI com `acima_de` transitiva, protocolos com camada, porta padrão IANA e
transporte (TCP/UDP), criptografia, faixas privadas da RFC 1918, equipamentos por camada,
meios físicos e topologias. HTTPS está como `HTTPSobreTLS` (ver sugestões).

### ia_computacao.txt
Áreas da IA, algoritmos supervisionados, não supervisionados e por reforço (disjuntos),
busca cega e informada, formas de raciocínio, marcos (Turing 1950, Dartmouth 1956, ELIZA
1966, Deep Blue 1997, AlphaGo 2016, Transformer 2017), linguagens com criador, ano e
tipagem, estruturas de dados e complexidade de ordenação e busca.

### geografia_brasil_pe.txt
Cinco regiões, 26 estados e DF com capital e região, fronteiras dos estados do Nordeste,
cinco mesorregiões de Pernambuco com municípios, rios, biomas e patrimônio (Olinda 1982,
Ilhas Atlânticas 2001, Frevo 2012). Exceções reais: Fernando de Noronha é distrito estadual
e Brasília não é município.

### sistema_solar.txt
Oito planetas em ordem, rochosos e gigantes, planetas anões da UAI, satélites principais.
Exceções reais: Vênus e Urano com rotação retrógrada, Mercúrio e Vênus sem satélite natural.

## Como carregar

No chat do Info:

```
carregar bases/animais.txt
carregar bases            (lê todos os .txt da pasta; o LEIAME.md é ignorado)
```

Em Python:

```python
import info_datainfo as info
d = info.Dialogo()
print("\n".join(d.carregar("bases/geografia_brasil_pe.txt")))
```

## Validação feita (22/10/2026)

- Cada base e a pasta inteira carregadas com `Dialogo.carregar`: nenhuma linha ignorada,
  nenhum erro de leitura, soma dos fatos por arquivo igual ao total da pasta (sem fusão
  indevida de nomes entre bases).
- `verificar_consistencia(resolver=False)` depois de pensar: zero conflitos em todas.
- Exceções bloqueiam as heranças erradas (por exemplo `Avestruz voa` bloqueado por
  `Avestruz não_tem_propriedade voa`).
- Deduções conferidas por amostra: `Recife fica_em Brasil`, `Pernambuco tem_capital Recife`,
  `Planejamento precede Descarte`, `Python origem_nacional
  Holandesa`, `Lua é_um SateliteNatural`, `Aplicacao acima_de Fisica`.
- Suíte existente: `python -m unittest discover -s tests` com 28 testes OK.

## Fontes

- IBGE: divisão regional, unidades da federação, mesorregiões geográficas (1989/1990; em
  2017 substituídas por regiões geográficas intermediárias).
- UNESCO: Lista do Patrimônio Mundial (Olinda, 1982; Ilhas Atlânticas Brasileiras, 2001) e
  Patrimônio Cultural Imaterial (Frevo, 2012).
- Redes: ISO/IEC 7498-1 (OSI), registro de portas da IANA, RFC 791, RFC 8200, RFC 1918,
  IEEE 802.3 e 802.11.
- Sistema solar: NASA Solar System Exploration; UAI, Resolução B5 (2006).
- IA e computação: Russell e Norvig; Turing (1950); proposta de Dartmouth (1955);
  Vaswani et al. (2017); sites oficiais de Python, Java e Linux.


## Sugestões para o código

1. **Herança de exceções.** A regra ontológica `heranca_de_propriedade` herda só
   `tem_propriedade`. Uma classe com `não_tem_propriedade` (como `Ratita não voa`) não passa
   isso às instâncias, e a instância herda "voa" da superclasse. As bases contornam com a
   regra `heranca_negativa` e com o fato negativo direto na instância. Sugestão: incluir a
   regra de herança negativa em `_regras_ontologicas` e dar prioridade à classe mais
   específica.
2. **Siglas tratadas como plural.** Com o interpretador ativo, `HTTPS` foi casado com `HTTP`
   (o "S" final virou plural), o que gerou conflito funcional na porta padrão. Sugestão:
   não aplicar a regra de plural a nomes todo em maiúsculas ou com menos de 5 letras.
3. **Regras com o mesmo nome em arquivos diferentes.** Várias bases declaram
   `heranca_negativa`; ao carregar a pasta, uma substitui a outra (são iguais, então não há
   efeito). Vale avisar quando uma regra de mesmo nome tiver corpo diferente.
4. **Literais numéricos.** Anos, portas e taxas foram escritos como entidades (`Ano1991`,
   `Porta443`, `Taxa20PorCento`) para não colidir com o tratamento de nomes. Um tipo literal
   (número ou data) permitiria comparar valores em regras.
