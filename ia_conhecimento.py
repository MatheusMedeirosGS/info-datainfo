# %% [1] Configuração e dependências
# Sistema de Conhecimento com Aprendizado (SCA)
# Evolução dos códigos 1 (regra sobre dicionário) e 2 (lista de triplas).
# Bibliotecas: numpy, networkx e matplotlib (já instaladas no Google Colab).

import math
import random
import re
import json
import itertools
from datetime import datetime, date
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

random.seed(42)
np.random.seed(42)

K_PRIOR = 10.0        # peso da crença inicial, em "evidências equivalentes"
LIMIAR_USO = 0.30     # confiança mínima para um fato participar de inferências
LIMIAR_ACEITE = 0.50  # confiança mínima para um fato ser tratado como verdadeiro
FONTES_FIRMES = ("observado", "confirmado")

# nome interno (sem acento) -> nome como a pessoa escreveu, só para mostrar na tela
EXIBICAO = {}

SINONIMOS_RELACAO = {
    "é_uma": "é_um", "e_um": "é_um", "e_uma": "é_um",
    "eh_um": "é_um", "eh_uma": "é_um", "is_a": "é_um",
}


def e_var(t):
    """Termos iniciados por '?' são variáveis de regra."""
    return isinstance(t, str) and t.startswith("?")


def fmt(t):
    return f"({t[0]} {t[1]} {t[2]})"


# %% [2] Estruturas de conhecimento: Fato, Regra e BaseConhecimento

class Fato:
    """Tripla (sujeito, relação, objeto) com crença Beta(alpha, beta)."""

    def __init__(self, s, r, o, confianca=1.0, fonte="observado", justificativa=None,
                 valido_ate=None):
        c = min(max(confianca, 0.01), 0.99)
        self.s, self.r, self.o = s, r, o
        self.alpha = c * K_PRIOR
        self.beta = (1 - c) * K_PRIOR
        self.fonte = fonte
        self.justificativa = justificativa  # (nome_regra, [chaves das premissas])
        self.procedencias = []              # de onde veio: arquivo, linha, conversa, quando
        self.valido_ate = valido_ate        # "AAAA-MM-DD" ou None (vale sem prazo)

    @property
    def chave(self):
        return (self.s, self.r, self.o)

    @property
    def confianca(self):
        return self.alpha / (self.alpha + self.beta)

    def reforcar(self, peso=1.0):
        self.alpha += peso

    def enfraquecer(self, peso=1.0):
        self.beta += peso

    def __repr__(self):
        return f"{self.s} --{self.r}--> {self.o} [{self.confianca:.2f} | {self.fonte}]"


class Regra:
    """Regra de Horn: premissa_1 E ... E premissa_n => conclusão, com crença Beta."""

    def __init__(self, nome, premissas, conclusao, confianca=0.9, origem="manual",
                 alpha=None, beta=None, metricas=None):
        c = min(max(confianca, 0.01), 0.99)
        self.nome = nome
        self.premissas = [tuple(p) for p in premissas]
        self.conclusao = tuple(conclusao)
        self.alpha = alpha if alpha is not None else c * K_PRIOR
        self.beta = beta if beta is not None else (1 - c) * K_PRIOR
        self.origem = origem
        self.metricas = metricas or {}
        self.disparos = 0

    @property
    def confianca(self):
        return self.alpha / (self.alpha + self.beta)

    def assinatura(self):
        """Forma canônica: independe do nome das variáveis e da ordem das premissas."""
        melhor = None
        for ordem in itertools.permutations(self.premissas):
            mapa = {}

            def ren(t):
                if e_var(t):
                    if t not in mapa:
                        mapa[t] = f"?v{len(mapa)}"
                    return mapa[t]
                return t

            conc = tuple(ren(t) for t in self.conclusao)
            prem = tuple(tuple(ren(t) for t in p) for p in ordem)
            cand = (conc, prem)
            if melhor is None or cand < melhor:
                melhor = cand
        return melhor

    def __str__(self):
        corpo = " E ".join(fmt(p) for p in self.premissas)
        return f"[{self.nome}] {corpo} => {fmt(self.conclusao)}  conf={self.confianca:.2f} ({self.origem})"


class BaseConhecimento:
    """Armazena fatos com índices por sujeito, relação e objeto, e a ontologia das relações."""

    def __init__(self):
        self.fatos = {}
        self._idx_s = defaultdict(set)
        self._idx_r = defaultdict(set)
        self._idx_o = defaultdict(set)
        self.transitivas = set()
        self.simetricas = set()
        self.inversas = {}
        self.funcionais = set()
        self.negacoes = {}      # relação positiva -> relação negativa
        self.disjuntas = []     # pares de classes que não podem coexistir
        self.rejeitados = set() # fatos refutados por feedback; nunca são rederivados
        self.bloqueados = set() # deduções barradas por conflito; recalculado a cada raciocínio
        self.penalizados = set()  # (perdedor, vencedor) já penalizados: cada conflito pesa uma vez
        self.formas = {}        # classe -> relações obrigatórias do cadastro
        self.vocabulario = {}   # frase normalizada -> relação (aprendido em conversa)
        self.rotulos = []       # julgamentos do usuário: (s, r, o, verdade)
        self.fontes_stats = {}  # fonte -> [acertos, erros] quando um conflito é decidido
        self.proc_atual = {"via": "conversa"}  # procedência dos fatos adicionados agora
        self.config = {}        # modos do raciocínio: conjuncao, acumulo, fatores, rotulos_fuzzy
        self.faq = []           # perguntas frequentes documentadas: (id, pergunta, resposta)

    @staticmethod
    def normalizar_relacao(r):
        r = r.strip().lower().replace(" ", "_")
        return SINONIMOS_RELACAO.get(r, r)

    # Ontologia das relações
    def declarar_transitiva(self, r):
        self.transitivas.add(self.normalizar_relacao(r))

    def declarar_simetrica(self, r):
        self.simetricas.add(self.normalizar_relacao(r))

    def declarar_inversa(self, r1, r2):
        self.inversas[self.normalizar_relacao(r1)] = self.normalizar_relacao(r2)

    def declarar_funcional(self, r):
        self.funcionais.add(self.normalizar_relacao(r))

    def declarar_negacao(self, positiva, negativa):
        self.negacoes[self.normalizar_relacao(positiva)] = self.normalizar_relacao(negativa)

    def declarar_disjuntas(self, classe_a, classe_b):
        if (classe_a, classe_b) not in self.disjuntas:
            self.disjuntas.append((classe_a, classe_b))

    def relacao_oposta(self, r):
        if r in self.negacoes:
            return self.negacoes[r]
        for pos, neg in self.negacoes.items():
            if neg == r:
                return pos
        return None

    # Confiabilidade das fontes (arquivos): começa neutra e só cai quando a fonte perde conflitos
    PRIOR_FONTE = (19.0, 1.0)

    def peso_fonte(self, fonte):
        if not fonte or fonte not in self.fontes_stats:
            return 1.0
        acertos, erros = self.fontes_stats[fonte]
        media = (self.PRIOR_FONTE[0] + acertos) / (sum(self.PRIOR_FONTE) + acertos + erros)
        return min(1.0, media / (self.PRIOR_FONTE[0] / sum(self.PRIOR_FONTE)))

    def registrar_fonte(self, fonte, acertou):
        if fonte and fonte != "conversa":
            par = self.fontes_stats.setdefault(fonte, [0, 0])
            par[0 if acertou else 1] += 1

    # Manipulação de fatos
    def adicionar(self, s, r, o, confianca=1.0, fonte="observado", justificativa=None,
                  valido_ate=None):
        r = self.normalizar_relacao(r)
        chave = (s, r, o)
        firme = fonte in FONTES_FIRMES
        proc = None
        if firme:
            proc = dict(self.proc_atual, quando=datetime.now().isoformat(timespec="seconds"))
            confianca = max(0.05, confianca * self.peso_fonte(proc.get("arquivo")))
        if chave in self.rejeitados:
            if not firme:
                return None
            self.rejeitados.discard(chave)
        f = self.fatos.get(chave)
        if f is not None and firme:
            self._anotar(f, proc, valido_ate)
        if f is not None:
            if firme and f.fonte not in FONTES_FIRMES:
                # fato que era só deduzido/estimado e agora foi observado: a crença
                # passa a ser a da observação (somar 1 voto ao prior sintético do fato
                # deduzido quase não mudava a confiança) e a justificativa antiga sai.
                base_nova = Fato(s, r, o, confianca, fonte)
                f.alpha, f.beta = base_nova.alpha, base_nova.beta
                f.fonte, f.justificativa = fonte, None
            elif firme:
                # nova observação soma evidência ponderada pela confiança informada
                f.alpha += confianca
                f.beta += 1 - confianca
            elif f.fonte not in FONTES_FIRMES and confianca > f.confianca:
                novo = Fato(s, r, o, confianca, fonte, justificativa)
                f.alpha, f.beta = novo.alpha, novo.beta
                f.fonte, f.justificativa = fonte, justificativa
            return f
        f = Fato(s, r, o, confianca, fonte, justificativa, valido_ate)
        if proc:
            f.procedencias.append(proc)
        self.fatos[chave] = f
        self._idx_s[s].add(chave)
        self._idx_r[r].add(chave)
        self._idx_o[o].add(chave)
        return f

    @staticmethod
    def _anotar(f, proc, valido_ate):
        """Registra de onde veio mais uma observação do mesmo fato (sem repetir a mesma origem)."""
        igual = lambda p: (p.get("arquivo"), p.get("linha"), p.get("via")) == (
            proc.get("arquivo"), proc.get("linha"), proc.get("via"))
        if not any(igual(p) for p in f.procedencias):
            f.procedencias.append(proc)
            del f.procedencias[:-6]
        if valido_ate:
            f.valido_ate = valido_ate

    def remover(self, chave):
        f = self.fatos.pop(chave, None)
        if f:
            self._idx_s[f.s].discard(chave)
            self._idx_r[f.r].discard(chave)
            self._idx_o[f.o].discard(chave)
        return f

    def consultar(self, s=None, r=None, o=None, minimo=0.0):
        conjuntos = []
        if s is not None:
            conjuntos.append(self._idx_s.get(s, set()))
        if r is not None:
            conjuntos.append(self._idx_r.get(self.normalizar_relacao(r), set()))
        if o is not None:
            conjuntos.append(self._idx_o.get(o, set()))
        if conjuntos:
            conjuntos.sort(key=len)
            chaves = set.intersection(*conjuntos)
        else:
            chaves = set(self.fatos)
        return [self.fatos[k] for k in sorted(chaves) if self.fatos[k].confianca >= minimo]

    def entidades(self):
        return sorted({f.s for f in self.fatos.values()} | {f.o for f in self.fatos.values()})


# %% [3] Motor de inferência: encadeamento para frente, bloqueio, consistência, explicação e abdução

class MotorInferencia:

    def __init__(self, base):
        self.base = base
        self.regras = []
        self._assinaturas = {}
        self.bloqueios = {}
        self.geracoes = []                # quantos fatos novos surgiram em cada rodada
        self.n_derivacoes = {}            # caminhos independentes até cada conclusão

    @property
    def modo_conjuncao(self):
        """"produto" (probabilidade) ou "minimo" (lógica fuzzy)."""
        return self.base.config.get("conjuncao", "produto")

    @property
    def modo_acumulo(self):
        """"maximo" (vale o melhor caminho) ou "ruido_ou" (evidências independentes se somam)."""
        return self.base.config.get("acumulo", "maximo")

    def _conf_regra(self, regra):
        """Confiança da regra, com o ajuste de grupo que a calibração (algoritmo genético) define."""
        fator = self.base.config.get("fatores", {})
        grupo = ("heranca" if regra.nome == "heranca_de_propriedade"
                 else "usuario" if regra.origem in ("manual", "induzida") else "outras")
        return min(0.99, regra.confianca * fator.get(grupo, 1.0))

    def _conjuncao(self, conf_regra, confs):
        if self.modo_conjuncao == "minimo":
            return min([conf_regra] + list(confs))
        return conf_regra * math.prod(confs)

    # Gestão de regras
    def tem_regra(self, regra):
        return regra.assinatura() in self._assinaturas

    def adicionar_regra(self, regra):
        """Adiciona a regra se ainda não existir equivalente. Retorna a regra armazenada."""
        a = regra.assinatura()
        if a in self._assinaturas:
            return self._assinaturas[a]
        self.regras.append(regra)
        self._assinaturas[a] = regra
        return regra

    def regra(self, nome):
        return next((r for r in self.regras if r.nome == nome), None)

    def compilar_propriedades(self):
        """Transforma propriedades declaradas das relações em regras estruturais."""
        b = self.base
        for r in sorted(b.transitivas):
            self.adicionar_regra(Regra(f"transitiva[{r}]", [("?a", r, "?b"), ("?b", r, "?c")],
                                       ("?a", r, "?c"), 0.95, "estrutural"))
        for r in sorted(b.simetricas):
            self.adicionar_regra(Regra(f"simetrica[{r}]", [("?a", r, "?b")], ("?b", r, "?a"),
                                       0.99, "estrutural"))
        for r1, r2 in sorted(b.inversas.items()):
            self.adicionar_regra(Regra(f"inversa[{r1}>{r2}]", [("?a", r1, "?b")], ("?b", r2, "?a"),
                                       0.99, "estrutural"))
            self.adicionar_regra(Regra(f"inversa[{r2}>{r1}]", [("?a", r2, "?b")], ("?b", r1, "?a"),
                                       0.99, "estrutural"))

    # Unificação
    @staticmethod
    def _substituir(padrao, b):
        return tuple(b.get(t, t) if e_var(t) else t for t in padrao)

    def _casar_padrao(self, padrao, b):
        s, r, o = self._substituir(padrao, b)
        fatos = self.base.consultar(None if e_var(s) else s, r, None if e_var(o) else o,
                                    minimo=LIMIAR_USO)
        for f in fatos:
            novo = dict(b)
            ok = True
            for termo, valor in ((s, f.s), (o, f.o)):
                if e_var(termo):
                    if termo in novo and novo[termo] != valor:
                        ok = False
                        break
                    novo[termo] = valor
            if ok:
                yield novo, f

    def _satisfazer(self, premissas, b=None, usados=None):
        b = {} if b is None else b
        usados = [] if usados is None else usados
        if not premissas:
            yield b, usados
            return
        for nb, f in self._casar_padrao(premissas[0], b):
            yield from self._satisfazer(premissas[1:], nb, usados + [f])

    @staticmethod
    def _unificar(padrao, alvo):
        """Unifica a conclusão de uma regra com um alvo; None no alvo é curinga."""
        b = {}
        for p, a in zip(padrao, alvo):
            if a is None:
                continue
            if e_var(p):
                if p in b and b[p] != a:
                    return None
                b[p] = a
            elif p != a:
                return None
        return b

    # Encadeamento para frente até o ponto fixo
    def executar(self, max_iter=25):
        self.compilar_propriedades()
        # inferências antigas são descartadas e recalculadas com as crenças atuais
        for chave in [k for k, f in self.base.fatos.items() if f.fonte == "inferido"]:
            self.base.remover(chave)
        self.bloqueios = {}
        self.geracoes = []
        self.n_derivacoes = {}
        for regra in self.regras:
            regra.disparos = 0
        total = 0
        for _ in range(max_iter):
            derivacoes = defaultdict(list)
            for regra in self.regras:
                if regra.confianca < LIMIAR_USO:
                    continue
                for b, usados in list(self._satisfazer(regra.premissas)):
                    s, r, o = self._substituir(regra.conclusao, b)
                    if e_var(s) or e_var(o) or s == o:
                        continue
                    conf = self._conjuncao(self._conf_regra(regra), [f.confianca for f in usados])
                    if conf < LIMIAR_USO:
                        continue
                    chave = (s, r, o)
                    if chave in self.base.rejeitados or chave in self.base.bloqueados:
                        continue
                    derivacoes[chave].append((conf, regra, [f.chave for f in usados]))
            candidatos = {}
            for chave, lista in derivacoes.items():
                melhor = max(lista, key=lambda d: d[0])
                if self.modo_acumulo == "ruido_ou":
                    # evidências independentes se somam: 1 - produto das dúvidas. Um caminho que
                    # passa por uma dedução apoiada na própria conclusão é circular e não conta.
                    livres = [d for d in lista if not any(self._depende(p, chave) for p in d[2])]
                    conf = min(0.99, 1 - math.prod(1 - d[0] for d in (livres or [melhor])))
                else:
                    conf = melhor[0]
                s, r, o = chave
                existente = self.base.fatos.get(chave)
                if existente and (existente.fonte != "inferido" or existente.confianca >= conf - 1e-9):
                    continue
                oposta = self.base.relacao_oposta(r)
                contra = self.base.fatos.get((s, oposta, o)) if oposta else None
                if contra and contra.confianca >= conf:
                    self.bloqueios[chave] = (contra.chave, melhor[1].nome)
                    continue
                candidatos[chave] = (conf, melhor[1], melhor[2])
                self.n_derivacoes[chave] = len(lista)
            if not candidatos:
                break
            novos = sum(1 for c in candidatos if c not in self.base.fatos)
            if novos:
                self.geracoes.append(novos)
            for (s, r, o), (conf, regra, prem) in candidatos.items():
                self.base.adicionar(s, r, o, conf, "inferido", (regra.nome, prem))
                regra.disparos += 1
                total += 1
        return total

    def _depende(self, premissa, alvo, profundidade=8):
        """True se a dedução `premissa` se apoia, direta ou indiretamente, em `alvo`."""
        f = self.base.fatos.get(premissa)
        if f is None or f.fonte != "inferido" or not f.justificativa or profundidade == 0:
            return False
        subs = f.justificativa[1]
        return alvo in subs or any(self._depende(q, alvo, profundidade - 1) for q in subs)

    # Consistência
    def _perde(self, perdedor, vencedor):
        """Aplica o resultado de um conflito. Fato deduzido é barrado (a dedução sai); fato
        informado perde confiança, mas só uma vez por conflito (senão cada raciocínio
        pioraria a mesma disputa)."""
        b = self.base
        if perdedor.fonte == "inferido":
            b.bloqueados.add(perdedor.chave)
            return
        par = (perdedor.chave, vencedor.chave)
        if par not in b.penalizados:
            b.penalizados.add(par)
            perdedor.enfraquecer(3.0)

    def verificar_consistencia(self, resolver=True):
        b = self.base
        conflitos = []
        for pos, neg in b.negacoes.items():
            for f in b.consultar(r=pos, minimo=LIMIAR_ACEITE):
                g = b.fatos.get((f.s, neg, f.o))
                if g and g.confianca >= LIMIAR_ACEITE:
                    perdedor = min((f, g), key=lambda x: x.confianca)
                    vencedor = g if perdedor is f else f
                    conflitos.append(("negação", f.chave, g.chave, perdedor.chave))
                    if resolver:
                        self._perde(perdedor, vencedor)
        for r in sorted(b.funcionais):
            por_sujeito = defaultdict(list)
            for f in b.consultar(r=r, minimo=LIMIAR_ACEITE):
                por_sujeito[f.s].append(f)
            for fs in por_sujeito.values():
                if len(fs) > 1:
                    fs.sort(key=lambda x: x.confianca, reverse=True)
                    for perdedor in fs[1:]:
                        conflitos.append(("funcional", fs[0].chave, perdedor.chave, perdedor.chave))
                        if resolver:
                            self._perde(perdedor, fs[0])
        for a, c in b.disjuntas:
            for f in b.consultar(r="é_um", o=a, minimo=LIMIAR_ACEITE):
                g = b.fatos.get((f.s, "é_um", c))
                if g and g.confianca >= LIMIAR_ACEITE:
                    conflitos.append(("disjunção", f.chave, g.chave, None))
        return conflitos

    # Explicação (árvore de prova)
    def explicar(self, s, r, o, nivel=0, visitados=None):
        visitados = set() if visitados is None else visitados
        pad = "    " * nivel
        chave = (s, self.base.normalizar_relacao(r), o)
        f = self.base.fatos.get(chave)
        if f is None:
            return [f"{pad}{fmt(chave)}: desconhecido"]
        linhas = [f"{pad}{fmt(chave)}  conf={f.confianca:.2f}  fonte={f.fonte}"]
        if f.justificativa and chave not in visitados and nivel < 8:
            visitados.add(chave)
            nome, premissas = f.justificativa
            regra = self.regra(nome)
            desc = f" (conf da regra {regra.confianca:.2f})" if regra else ""
            linhas.append(f"{pad}  porque a regra '{nome}'{desc} usou:")
            for p in premissas:
                linhas += self.explicar(*p, nivel=nivel + 1, visitados=visitados)
        return linhas

    # Abdução: quais fatos faltam para concluir algo
    def abduzir(self, s=None, r=None, o=None, limite=10):
        alvo = (s, self.base.normalizar_relacao(r) if r else None, o)
        ancoras = {t for t in (s, o) if t is not None}
        resultados = {}
        for regra in self.regras:
            if len(regra.premissas) < 2 or regra.confianca < LIMIAR_USO:
                continue
            b0 = self._unificar(regra.conclusao, alvo)
            if b0 is None:
                continue
            for i, premissa in enumerate(regra.premissas):
                outras = regra.premissas[:i] + regra.premissas[i + 1:]
                for b, usados in itertools.islice(self._satisfazer(outras, dict(b0)), 20):
                    if ancoras and not any(a in (f.s, f.o) for f in usados for a in ancoras):
                        continue
                    if next(self._casar_padrao(premissa, b), None) is not None:
                        continue
                    faltante = self._substituir(premissa, b)
                    if faltante in self.base.rejeitados:
                        continue  # já foi perguntado e recusado antes; não insistir
                    conclusao = self._substituir(regra.conclusao, b)
                    existente = self.base.fatos.get(conclusao)
                    if existente and existente.confianca >= LIMIAR_ACEITE:
                        continue
                    k = (faltante, conclusao)
                    if k not in resultados or regra.confianca > resultados[k][0]:
                        resultados[k] = (regra.confianca, regra.nome, faltante, conclusao)
        return sorted(resultados.values(), key=lambda x: -x[0])[:limite]


# %% [4] Aprendizado: indução de regras, embeddings TransE e classificador de conceitos

class IndutorDeRegras:
    """Mineração de regras sobre caminhos de até 2 saltos (inspirado no AMIE).

    Corpo candidato: (?x r1 ?y) ou (?x r1 ?z) E (?z r2 ?y), com r1 e r2 em qualquer
    direção. Métricas: suporte = |corpo ∩ cabeça|, confiança = suporte / |corpo|,
    cobertura = suporte / |cabeça|. A crença inicial da regra é Beta(suporte+1, erros+1).
    """

    def __init__(self, base, motor, suporte_min=3, conf_min=0.6, max_por_relacao=3,
                 ignorar=("subclasse_de",)):
        self.base = base
        self.motor = motor
        self.suporte_min = suporte_min
        self.conf_min = conf_min
        self.max_por_relacao = max_por_relacao
        self.ignorar = set(ignorar)
        self.contador = 0

    def _pares(self):
        ignorar = self.ignorar | set(self.base.negacoes.values())
        pares = defaultdict(set)
        for f in self.base.fatos.values():
            if f.fonte in FONTES_FIRMES and f.confianca >= LIMIAR_ACEITE and f.r not in ignorar:
                pares[f.r].add((f.s, f.o))
        return pares

    @staticmethod
    def _atomo(r, d, a, b):
        return (a, r, b) if d == "+" else (b, r, a)

    def _avaliar(self, cabeca, H, corpo, B, saida):
        if not B:
            return
        suporte = len(B & H)
        if suporte < self.suporte_min:
            return
        conf = suporte / len(B)
        if conf < self.conf_min:
            return
        saida.append(Regra(
            "candidata", corpo, ("?x", cabeca, "?y"), origem="induzida",
            alpha=suporte + 1, beta=len(B) - suporte + 1,
            metricas={"suporte": suporte, "corpo": len(B), "conf_padrao": round(conf, 3),
                      "cobertura": round(suporte / len(H), 3)}))

    def induzir(self):
        pares = self._pares()
        rels = sorted(pares)
        orientado = {}
        adjacencia = {}
        for r in rels:
            orientado[(r, "+")] = pares[r]
            orientado[(r, "-")] = {(o, s) for s, o in pares[r]}
            for d in "+-":
                m = defaultdict(set)
                for a, c in orientado[(r, d)]:
                    m[a].add(c)
                adjacencia[(r, d)] = m

        novas = []
        for cabeca in rels:
            H = pares[cabeca]
            if len(H) < self.suporte_min:
                continue
            candidatas = []
            for r1, d1 in itertools.product(rels, "+-"):
                if not (r1 == cabeca and d1 == "+"):
                    self._avaliar(cabeca, H, [self._atomo(r1, d1, "?x", "?y")],
                                  orientado[(r1, d1)], candidatas)
                for r2, d2 in itertools.product(rels, "+-"):
                    m = adjacencia[(r2, d2)]
                    B = {(x, y) for x, z in orientado[(r1, d1)] for y in m.get(z, ()) if x != y}
                    corpo = [self._atomo(r1, d1, "?x", "?z"), self._atomo(r2, d2, "?z", "?y")]
                    self._avaliar(cabeca, H, corpo, B, candidatas)
            aceitas = 0
            for c in sorted(candidatas, key=lambda r: (-r.confianca, -r.metricas["suporte"])):
                if aceitas >= self.max_por_relacao:
                    break
                if not self.motor.tem_regra(c):
                    self.contador += 1
                    c.nome = f"induzida_{self.contador}"
                    self.motor.adicionar_regra(c)
                    novas.append(c)
                    aceitas += 1
        return novas


class EmbeddingsTransE:
    """Embeddings de grafo TransE: para um fato verdadeiro, vetor(s) + vetor(r) ≈ vetor(o).

    Treino por ranking com margem: dist(positivo) + margem < dist(negativo),
    onde o negativo troca aleatoriamente o sujeito ou o objeto.
    """

    def __init__(self, dim=32, margem=1.0, taxa=0.02, epocas=300, semente=42):
        self.dim, self.margem, self.taxa, self.epocas = dim, margem, taxa, epocas
        self.rng = np.random.default_rng(semente)
        self.perdas = []

    def treinar(self, base):
        triplas = [f.chave for f in base.fatos.values() if f.confianca >= LIMIAR_ACEITE]
        self.ents = sorted({t[0] for t in triplas} | {t[2] for t in triplas})
        self.rels = sorted({t[1] for t in triplas})
        self.ie = {e: i for i, e in enumerate(self.ents)}
        self.ir = {r: i for i, r in enumerate(self.rels)}
        lim = 6 / math.sqrt(self.dim)
        self.E = self.rng.uniform(-lim, lim, (len(self.ents), self.dim))
        self.R = self.rng.uniform(-lim, lim, (len(self.rels), self.dim))
        self.E /= np.linalg.norm(self.E, axis=1, keepdims=True)
        self.R /= np.linalg.norm(self.R, axis=1, keepdims=True)
        dados = np.array([(self.ie[s], self.ir[r], self.ie[o]) for s, r, o in triplas])
        conhecidos = {tuple(x) for x in dados.tolist()}
        n = len(self.ents)
        self.perdas = []
        for _ in range(self.epocas):
            total = 0.0
            for h, r, t in dados[self.rng.permutation(len(dados))]:
                if self.rng.random() < 0.5:
                    h2, t2 = int(self.rng.integers(n)), t
                else:
                    h2, t2 = h, int(self.rng.integers(n))
                if (h2, r, t2) in conhecidos:
                    continue
                ep = self.E[h] + self.R[r] - self.E[t]
                en = self.E[h2] + self.R[r] - self.E[t2]
                perda = self.margem + ep @ ep - en @ en
                if perda > 0:
                    total += perda
                    gp, gn = 2 * ep, 2 * en
                    self.E[h] -= self.taxa * gp
                    self.E[t] += self.taxa * gp
                    self.R[r] -= self.taxa * (gp - gn)
                    self.E[h2] += self.taxa * gn
                    self.E[t2] -= self.taxa * gn
            self.E /= np.linalg.norm(self.E, axis=1, keepdims=True)
            self.perdas.append(total)
        self._triplas = triplas
        return self

    def distancia(self, s, r, o):
        if s not in self.ie or o not in self.ie or r not in self.ir:
            return None
        v = self.E[self.ie[s]] + self.R[self.ir[r]] - self.E[self.ie[o]]
        return float(np.linalg.norm(v))

    def sugerir(self, base, relacoes, top_k=10):
        """Prevê fatos ausentes. Plausibilidade = sigmoide calibrada pela distância
        média dos fatos conhecidos versus a dos candidatos da mesma relação."""
        sugestoes = []
        for r in relacoes:
            conhecidas = [t for t in self._triplas if t[1] == r]
            if not conhecidas:
                continue
            cabecas = sorted({t[0] for t in conhecidas})
            caudas = sorted({t[2] for t in conhecidas})
            d_pos = np.mean([self.distancia(*t) for t in conhecidas])
            cands = [(s, r, o) for s in cabecas for o in caudas
                     if s != o and (s, r, o) not in base.fatos and (s, r, o) not in base.rejeitados]
            if not cands:
                continue
            dists = [self.distancia(*c) for c in cands]
            d_neg = float(np.mean(dists))
            limiar = (d_pos + d_neg) / 2
            escala = max((d_neg - d_pos) / 4, 1e-6)
            for c, d in zip(cands, dists):
                p = 1 / (1 + math.exp(-(limiar - d) / escala))
                sugestoes.append((p, d, c))
        sugestoes.sort(key=lambda x: -x[0])
        return sugestoes[:top_k]


class EmbeddingsRotatE(EmbeddingsTransE):
    """Embeddings RotatE: cada relação é uma rotação no plano complexo, e(h) * rot(r) ≈ e(t).

    Ao contrário do TransE, consegue representar relações simétricas (rotação de 180 graus) e
    inversas (rotações opostas). Mesma interface do TransE: treinar, distancia e sugerir.
    """

    def __init__(self, dim=32, margem=1.0, taxa=0.05, epocas=300, semente=42):
        super().__init__(dim, margem, taxa, epocas, semente)

    def treinar(self, base):
        triplas = [f.chave for f in base.fatos.values() if f.confianca >= LIMIAR_ACEITE]
        self.ents = sorted({t[0] for t in triplas} | {t[2] for t in triplas})
        self.rels = sorted({t[1] for t in triplas})
        self.ie = {e: i for i, e in enumerate(self.ents)}
        self.ir = {r: i for i, r in enumerate(self.rels)}
        lim = 1 / math.sqrt(self.dim)
        self.Er = self.rng.uniform(-lim, lim, (len(self.ents), self.dim))
        self.Ei = self.rng.uniform(-lim, lim, (len(self.ents), self.dim))
        self.Th = self.rng.uniform(-math.pi, math.pi, (len(self.rels), self.dim))
        dados = np.array([(self.ie[s], self.ir[r], self.ie[o]) for s, r, o in triplas])
        conhecidos = {tuple(x) for x in dados.tolist()}
        n = len(self.ents)
        self.perdas = []
        for _ in range(self.epocas):
            total = 0.0
            for h, r, t in dados[self.rng.permutation(len(dados))]:
                if self.rng.random() < 0.5:
                    h2, t2 = int(self.rng.integers(n)), t
                else:
                    h2, t2 = h, int(self.rng.integers(n))
                if (h2, r, t2) in conhecidos:
                    continue
                c, s = np.cos(self.Th[r]), np.sin(self.Th[r])
                dpr, dpi = self._resto(h, t, c, s)
                dnr, dni = self._resto(h2, t2, c, s)
                perda = self.margem + (dpr @ dpr + dpi @ dpi) - (dnr @ dnr + dni @ dni)
                if perda > 0:
                    total += perda
                    self._passo(h, t, r, c, s, dpr, dpi, +1.0)
                    self._passo(h2, t2, r, c, s, dnr, dni, -1.0)
            norma = np.sqrt(self.Er ** 2 + self.Ei ** 2).sum(axis=1, keepdims=True) / self.dim
            norma = np.maximum(norma, 1e-9)
            self.Er, self.Ei = self.Er / norma, self.Ei / norma
            self.perdas.append(total)
        self._triplas = triplas
        return self

    def _resto(self, h, t, c, s):
        """Parte real e imaginária de e(h) * rot(r) - e(t)."""
        hr, hi = self.Er[h], self.Ei[h]
        return hr * c - hi * s - self.Er[t], hr * s + hi * c - self.Ei[t]

    def _passo(self, h, t, r, c, s, dr, di, sinal):
        """Um passo de gradiente do quadrado da distância (sinal +1 aproxima, -1 afasta)."""
        k = self.taxa * sinal
        hr, hi = self.Er[h].copy(), self.Ei[h].copy()
        grad_th = 2 * (dr * (-hr * s - hi * c) + di * (hr * c - hi * s))
        self.Er[h] -= k * 2 * (dr * c + di * s)
        self.Ei[h] -= k * 2 * (-dr * s + di * c)
        self.Er[t] += k * 2 * dr
        self.Ei[t] += k * 2 * di
        self.Th[r] -= k * grad_th

    def distancia(self, s, r, o):
        if s not in self.ie or o not in self.ie or r not in self.ir:
            return None
        h, t, k = self.ie[s], self.ie[o], self.ir[r]
        dr, di = self._resto(h, t, np.cos(self.Th[k]), np.sin(self.Th[k]))
        return float(np.sqrt(dr @ dr + di @ di))


class ClassificadorConceitos:
    """Naive Bayes de Bernoulli: aprende a classe (é_um) a partir dos atributos da entidade.

    Atributo = par (relação, objeto) ligado à entidade, por exemplo
    ('tem_propriedade', 'penas'). Suavização de Laplace.

    Por padrão, só usa `tem_propriedade`/`não_tem_propriedade` como atributo (as
    características físicas, igual ao código 1 original). Isso é proposital: numa base
    com relações relacionais além de propriedades (por exemplo `predador_de`,
    `compete_com`, num domínio de ecologia), deixar o classificador "ver" quem come quem
    é ruído para a pergunta que ele responde ("isso é fisicamente parecido com uma ave,
    um mamífero ou um peixe?"), e distorcia a confiança na prática (testado: uma
    entidade com um único fato físico chegava a mais de 80% de confiança só porque
    relações ecológicas de outras entidades inflavam o vocabulário).
    """

    def __init__(self, base, relacoes=("tem_propriedade", "não_tem_propriedade")):
        self.base = base
        self.relacoes = set(relacoes)
        self.classes = []  # fica vazio até treinar(); evita AttributeError se usado antes

    def atributos(self, entidade):
        return {(f.r, f.o) for f in self.base.consultar(s=entidade, minimo=LIMIAR_ACEITE)
                if f.r in self.relacoes}

    def treinar(self, classes):
        self.classes = list(classes)
        self.exemplos = defaultdict(list)
        for f in self.base.consultar(r="é_um"):
            if f.fonte in FONTES_FIRMES and f.o in self.classes:
                self.exemplos[f.o].append(f.s)
        self.vocab = sorted({a for ents in self.exemplos.values() for e in ents
                             for a in self.atributos(e)})
        total = sum(len(v) for v in self.exemplos.values())
        self.log_prior = {c: math.log((len(self.exemplos[c]) + 1) / (total + len(self.classes)))
                          for c in self.classes}
        self.p = {}
        for c in self.classes:
            ents = self.exemplos[c]
            cont = Counter(a for e in ents for a in self.atributos(e))
            self.p[c] = {a: (cont[a] + 1) / (len(ents) + 2) for a in self.vocab}
        return self

    def classificar(self, entidade):
        attrs = self.atributos(entidade)
        scores = {}
        for c in self.classes:
            s = self.log_prior[c]
            for a in self.vocab:
                p = self.p[c][a]
                s += math.log(p if a in attrs else 1 - p)
            scores[c] = s
        m = max(scores.values())
        exp = {c: math.exp(v - m) for c, v in scores.items()}
        z = sum(exp.values())
        return sorted(((c, v / z) for c, v in exp.items()), key=lambda x: -x[1])


# %% [5] Interface em linguagem natural (português controlado)

class InterfaceLinguagem:
    AJUDA = ("Formatos aceitos:\n"
             "  Gato predador_de Rato   -> ensina um fato\n"
             "  Gato predador_de Rato?  -> verifica um fato (no chat, pergunta de volta se não souber)\n"
             "  quem predador_de Salmao? -> busca sujeitos (pergunta de volta se não souber)\n"
             "  o que Gato predador_de? -> busca objetos (pergunta de volta se não souber)\n"
             "  por que Cachorro evoluiu_de Carnivoro?\n"
             "  o que falta para Criatura?\n"
             "  classifique Misterio\n"
             "  errado Gato predador_de Golfinho / certo Cobra predador_de Rato\n"
             "  ensine Avestruz         -> modo entrevista: a IA pergunta e aprende sobre uma entidade nova\n"
             "  ensine                  -> mesma coisa, mas a IA pergunta o nome primeiro")

    GATILHOS_ENTREVISTA = ("ensine", "ensinar", "entrevista", "nova entidade",
                           "cadastrar entidade", "quero te ensinar")

    def __init__(self, ia):
        self.ia = ia

    def _ent(self, token):
        return self.ia._normalizar_entidade(token)

    def _gatilho_entrevista(self, t):
        """Se `t` começa com um gatilho de entrevista, devolve o nome que vier depois
        dele (ou "" se não vier nenhum). Senão, devolve None (não é esse comando)."""
        baixo = t.rstrip("?").strip().lower()
        for g in self.GATILHOS_ENTREVISTA:
            if baixo == g:
                return ""
            if baixo.startswith(g + " "):
                return t[len(g):].strip()
        return None

    def processar(self, texto, interativo=False):
        t = re.sub(r"\s+", " ", texto.strip())
        nome = self._gatilho_entrevista(t)
        if nome is not None:
            if not interativo:
                return ("O modo entrevista precisa de perguntas e respostas em sequência: "
                        "use o prompt (célula 20) ou prompt_terminal().")
            return self.ia.entrevistar(nome or None)
        pergunta = t.endswith("?")
        t = t.rstrip("?").strip()
        t = re.sub(r"\b(é|e|eh) (um|uma)\b", "é_um", t, flags=re.IGNORECASE)
        tk = t.split()
        low = [x.lower() for x in tk]
        rel = BaseConhecimento.normalizar_relacao
        if not tk:
            return self.AJUDA

        if low[:2] in (["por", "que"],) and len(tk) == 5:
            return "\n".join(self.ia.motor.explicar(self._ent(tk[2]), rel(tk[3]), self._ent(tk[4])))
        if low[0] in ("porque", "explique") and len(tk) == 4:
            return "\n".join(self.ia.motor.explicar(self._ent(tk[1]), rel(tk[2]), self._ent(tk[3])))
        if low[:3] == ["o", "que", "falta"] and len(tk) == 5:
            return self.ia.o_que_falta(self._ent(tk[4]), imprimir=False)
        if low[0] in ("classifique", "classificar") and len(tk) == 2:
            return self.ia.classificar(self._ent(tk[1]), imprimir=False)
        if low[0] in ("certo", "errado") and len(tk) == 4:
            s, r, o = self._ent(tk[1]), rel(tk[2]), self._ent(tk[3])
            return self.ia.feedback(s, r, o, low[0] == "certo", imprimir=False)
        if low[0] == "quem" and len(tk) == 3:
            r, o = rel(tk[1]), self._ent(tk[2])
            return self.ia.resolver_quem(r, o, interativo=interativo)
        if low[:2] == ["o", "que"] and len(tk) == 4:
            s, r = self._ent(tk[2]), rel(tk[3])
            return self.ia.resolver_o_que(s, r, interativo=interativo)
        if len(tk) == 3:
            s, r, o = self._ent(tk[0]), rel(tk[1]), self._ent(tk[2])
            if pergunta:
                return self.ia.resolver_fato(s, r, o, interativo=interativo)
            antes = len(self.ia.base.fatos)
            self.ia.base.adicionar(s, r, o, 1.0, "observado")
            self.ia.pensar(verbose=False)
            return (f"Aprendido: {fmt((s, r, o))}. A base passou de {antes} "
                    f"para {len(self.ia.base.fatos)} fatos.")
        return self.AJUDA


# %% [6] Fachada: SistemaIA (ciclo perceber, raciocinar, aprender, explicar)

class SistemaIA:

    def __init__(self):
        self.base = BaseConhecimento()
        self.motor = MotorInferencia(self.base)
        self.indutor = IndutorDeRegras(self.base, self.motor)
        self.embeddings = None
        self.classificador = ClassificadorConceitos(self.base)
        self.linguagem = InterfaceLinguagem(self)
        self._regras_ontologicas()

    def _regras_ontologicas(self):
        self.base.declarar_transitiva("subclasse_de")
        self.base.declarar_negacao("tem_propriedade", "não_tem_propriedade")
        for regra in [
            Regra("heranca_de_classe", [("?x", "é_um", "?c"), ("?c", "subclasse_de", "?d")],
                  ("?x", "é_um", "?d"), 0.98, "ontologica"),
            Regra("heranca_de_propriedade", [("?x", "é_um", "?c"), ("?c", "tem_propriedade", "?p")],
                  ("?x", "tem_propriedade", "?p"), 0.90, "ontologica"),
            Regra("propriedade_da_superclasse",
                  [("?c", "subclasse_de", "?d"), ("?d", "tem_propriedade", "?p")],
                  ("?c", "tem_propriedade", "?p"), 0.90, "ontologica"),
        ]:
            self.motor.adicionar_regra(regra)

    # Percepção
    def observar(self, triplas, confianca=1.0, fonte="observado"):
        for t in triplas:
            s, r, o = t[:3]
            c = t[3] if len(t) > 3 else confianca
            self.base.adicionar(s, r, o, c, fonte)

    @staticmethod
    def dicionario_para_triplas(nome, atributos):
        """Converte o formato do código 1 ({'tem_penas': True, 'voa': False}) em triplas."""
        triplas = []
        for chave, valor in atributos.items():
            prop = chave[4:] if chave.startswith("tem_") else chave
            rel = "tem_propriedade" if valor else "não_tem_propriedade"
            triplas.append((nome, rel, prop))
        return triplas

    # Raciocínio
    def pensar(self, verbose=True):
        self.base.bloqueados = set()
        n = self.motor.executar()
        conflitos = self.motor.verificar_consistencia(resolver=True)
        if conflitos:
            n = self.motor.executar()
        if verbose:
            cont = Counter(f.fonte for f in self.base.fatos.values())
            print(f"Raciocínio concluído: {n} fatos inferidos | base: {dict(cont)}")
            if self.motor.bloqueios:
                print(f"Inferências bloqueadas por exceção: {len(self.motor.bloqueios)}")
                for chave, (contra, regra) in sorted(self.motor.bloqueios.items()):
                    print(f"  {fmt(chave)} bloqueado por {fmt(contra)} (regra {regra})")
            for tipo, a, b, perdedor in conflitos:
                extra = f" -> crença reduzida em {fmt(perdedor)}" if perdedor else ""
                print(f"  Conflito de {tipo}: {fmt(a)} x {fmt(b)}{extra}")
        return n

    def aprender_regras(self, verbose=True):
        novas = self.indutor.induzir()
        if verbose:
            print(f"Regras induzidas a partir dos dados: {len(novas)}")
            for r in novas:
                print(f"  {r}\n      métricas: {r.metricas}")
        self.pensar(verbose=verbose)
        return novas

    def feedback(self, s, r, o, correto, imprimir=True, peso=1.0):
        """Aprendizado por reforço simbólico: ajusta a crença do fato e da regra que o gerou.
        `peso` abaixo de 1 vem de respostas incertas ("provavelmente"): ajusta menos e não descarta."""
        r = self.base.normalizar_relacao(r)
        chave = (s, r, o)
        f = self.base.fatos.get(chave)
        regra = self.motor.regra(f.justificativa[0]) if f and f.justificativa else None
        if peso < 1.0 and f is not None:
            if correto:
                f.reforcar(3.0 * peso)
                if regra:
                    regra.alpha += peso
            else:
                f.enfraquecer(3.0 * peso)
                if regra:
                    regra.beta += peso
        elif correto:
            if f:
                f.reforcar(3.0)
                f.fonte = "confirmado"
            else:
                self.base.adicionar(s, r, o, 1.0, "confirmado")
            if regra:
                regra.alpha += 1
        else:
            # Exceção a uma regra ontológica (herança) ou estrutural não é erro da
            # regra, então só regras manuais ou induzidas perdem confiança.
            if regra and regra.origem not in ("ontologica", "estrutural"):
                regra.beta += 1
            self.base.rejeitados.add(chave)
            self.base.remover(chave)
            oposta = self.base.relacao_oposta(r)
            if oposta:
                self.base.adicionar(s, oposta, o, 1.0, "observado")
        self.pensar(verbose=False)
        msg = f"Feedback {'positivo' if correto else 'negativo'} em {fmt(chave)}."
        if regra:
            msg += f" Regra '{regra.nome}' agora com conf {regra.confianca:.2f}."
        if imprimir:
            print(msg)
        return msg

    # Consulta
    def verificar(self, s, r, o):
        r = self.base.normalizar_relacao(r)
        f = self.base.fatos.get((s, r, o))
        if f and f.confianca >= LIMIAR_ACEITE:
            return f"Sim. {fmt(f.chave)} conf={f.confianca:.2f} ({f.fonte})"
        oposta = self.base.relacao_oposta(r)
        g = self.base.fatos.get((s, oposta, o)) if oposta else None
        if g and g.confianca >= LIMIAR_ACEITE:
            return f"Não. Existe o fato contrário {fmt(g.chave)} conf={g.confianca:.2f}"
        if (s, r, o) in self.base.rejeitados:
            return f"Não. {fmt((s, r, o))} foi refutado por feedback."
        linhas = [f"Não sei se {fmt((s, r, o))}. Preciso de mais fatos."]
        if self.embeddings:
            d = self.embeddings.distancia(s, r, o)
            if d is not None:
                linhas.append(f"  Intuição (embeddings): distância {d:.2f}")
        for conf, nome, falt, _ in self.motor.abduzir(s, r, o, limite=3):
            linhas.append(f"  Se eu soubesse {fmt(falt)}, concluiria pela regra '{nome}' ({conf:.2f})")
        return "\n".join(linhas)

    def _normalizar_entidade(self, token):
        mapa = {e.lower(): e for e in self.base.entidades()}
        return mapa.get(token.lower(), token)

    # IMPORTANTE: não incluir "não"/"nao"/"n" aqui. Em `fluxo_resolucao_fato`, "não" é
    # uma resposta própria (rejeita a premissa) e precisa continuar distinta de "não
    # sei" (pula a pergunta sem decidir nada). Já misturaram os dois antes e isso fazia
    # qualquer "não" ser tratado como "não sei" silenciosamente, bug real, encontrado
    # numa revisão de código e corrigido aqui.
    NAO_SEI = ("não sei", "nao sei", "nsei", "sei lá", "sei la", "nenhuma ideia")

    # Retroalimentação: quando falta informação, a IA pergunta de volta até entender.
    # Implementada como GERADOR: o método "pausa" num yield toda vez que precisa de uma
    # resposta seguinte, e "acorda" quando alguém manda essa resposta com .send(...).
    # Isso permite reaproveitar exatamente a mesma lógica em dois lugares diferentes:
    # o chat de texto (que pergunta com input()) e o chat visual com botões (seção 7),
    # que não pode usar input() porque não há como "pausar" a execução de uma célula
    # esperando clique de botão.
    def fluxo_resolucao_fato(self, s, r, o, max_rodadas=4):
        r = self.base.normalizar_relacao(r)
        resposta = self.verificar(s, r, o)
        if resposta.startswith("Sim.") or resposta.startswith("Não."):
            return resposta
        perguntados = set()
        rodada = 0
        while rodada < max_rodadas:
            candidatos = [c for c in self.motor.abduzir(s, r, o, limite=8)
                         if c[2] not in perguntados]
            if not candidatos:
                break
            conf, nome, faltante, _ = candidatos[0]
            perguntados.add(faltante)
            rodada += 1
            resp = (yield (f"Ainda não concluí se {fmt((s, r, o))}. Para ajudar: "
                          f"{fmt(faltante)} é verdade? (sim / não / não sei)")).strip().lower()
            fs, fr, fo = faltante
            if resp in ("sim", "s"):
                self.base.adicionar(fs, fr, fo, 1.0, "observado")
                self.pensar(verbose=False)
            elif resp in self.NAO_SEI:
                continue  # tenta outra premissa candidata na próxima rodada
            else:
                self.base.rejeitados.add(faltante)
                self.base.remover(faltante)
                self.pensar(verbose=False)
            resposta = self.verificar(s, r, o)
            if resposta.startswith("Sim.") or resposta.startswith("Não."):
                return resposta
        return (resposta + "\n  Mesmo perguntando, não reuni informação suficiente. "
                "Diga o fato diretamente (ex.: 'Gato predador_de Rato') quando souber.")

    def fluxo_resolucao_quem(self, r, o):
        r = self.base.normalizar_relacao(r)
        fs = self.base.consultar(r=r, o=o, minimo=LIMIAR_ACEITE)
        if fs:
            return "\n".join(f"  {f.s}  (conf {f.confianca:.2f}, {f.fonte})" for f in fs)
        resp = (yield (f"Não sei quem {r} {o}. Você sabe? "
                      f"(digite o nome, ou 'não'/'não sei' para eu desistir)")).strip()
        # Aqui (pergunta de uma resposta só) "não" também significa desistir, não é
        # ambíguo com outra coisa como em fluxo_resolucao_fato.
        if not resp or resp.lower() in self.NAO_SEI or resp.lower() in ("não", "nao", "n"):
            return f"Tudo bem, vou continuar sem saber quem {r} {o}."
        sujeito = self._normalizar_entidade(resp)
        self.base.adicionar(sujeito, r, o, 1.0, "observado")
        self.pensar(verbose=False)
        fs = self.base.consultar(r=r, o=o, minimo=LIMIAR_ACEITE)
        return "Aprendido: " + "; ".join(fmt(f.chave) for f in fs) + ". Obrigado por me ensinar!"

    def fluxo_resolucao_o_que(self, s, r):
        r = self.base.normalizar_relacao(r)
        fs = self.base.consultar(s=s, r=r, minimo=LIMIAR_ACEITE)
        if fs:
            return "\n".join(f"  {f.o}  (conf {f.confianca:.2f}, {f.fonte})" for f in fs)
        resp = (yield (f"Não sei o que {s} {r}. Você sabe? "
                      f"(digite a resposta, ou 'não'/'não sei' para eu desistir)")).strip()
        if not resp or resp.lower() in self.NAO_SEI or resp.lower() in ("não", "nao", "n"):
            return f"Tudo bem, vou continuar sem saber o que {s} {r}."
        objeto = self._normalizar_entidade(resp)
        self.base.adicionar(s, r, objeto, 1.0, "observado")
        self.pensar(verbose=False)
        fs = self.base.consultar(s=s, r=r, minimo=LIMIAR_ACEITE)
        return "Aprendido: " + "; ".join(fmt(f.chave) for f in fs) + ". Obrigado por me ensinar!"

    @staticmethod
    def _rodar_gerador_com_input(gerador, formatar_pergunta):
        """Aciona um 'fluxo_resolucao_*' usando input() até ele terminar. Usado pelo
        chat de texto. Devolve a resposta final (string)."""
        try:
            pergunta = next(gerador)
            while True:
                resp = input(formatar_pergunta(pergunta))
                pergunta = gerador.send(resp)
        except StopIteration as fim:
            return fim.value

    # As três funções abaixo são a "porta de entrada" usada pelo chat de texto
    # (InterfaceLinguagem.processar) e pelas demos: por fora, continuam se comportando
    # como antes (recebem s, r, o e devolvem uma string). Por dentro, só acionam o
    # gerador correspondente com input().
    def resolver_fato(self, s, r, o, interativo=False, max_rodadas=4):
        """Verifica (s r o). Se a confiança não bastar e interativo=True, usa a
        abdução para descobrir qual premissa falta e pergunta ao usuário sobre ela,
        repetindo até concluir, esgotar as rodadas ou o usuário não saber mais nada."""
        if not interativo:
            r = self.base.normalizar_relacao(r)
            return self.verificar(s, r, o)
        gen = self.fluxo_resolucao_fato(s, r, o, max_rodadas)
        return self._rodar_gerador_com_input(gen, lambda p: f"  [IA] {p} ")

    def resolver_quem(self, r, o, interativo=False):
        """Responde 'quem r o?'. Sem interativo, só relata que faltam fatos."""
        if not interativo:
            r = self.base.normalizar_relacao(r)
            fs = self.base.consultar(r=r, o=o, minimo=LIMIAR_ACEITE)
            if fs:
                return "\n".join(f"  {f.s}  (conf {f.confianca:.2f}, {f.fonte})" for f in fs)
            return f"Não sei quem {r} {o}. Preciso de mais fatos."
        gen = self.fluxo_resolucao_quem(r, o)
        return self._rodar_gerador_com_input(gen, lambda p: f"  [IA] {p} ")

    def resolver_o_que(self, s, r, interativo=False):
        """Responde 'o que s r?'. Sem interativo, só relata que faltam fatos."""
        if not interativo:
            r = self.base.normalizar_relacao(r)
            fs = self.base.consultar(s=s, r=r, minimo=LIMIAR_ACEITE)
            if fs:
                return "\n".join(f"  {f.o}  (conf {f.confianca:.2f}, {f.fonte})" for f in fs)
            return f"Não sei o que {s} {r}. Preciso de mais fatos."
        gen = self.fluxo_resolucao_o_que(s, r)
        return self._rodar_gerador_com_input(gen, lambda p: f"  [IA] {p} ")

    # --- Modo Entrevista: a IA se apresenta como especialista no domínio que já conhece
    # e PERGUNTA para extrair conhecimento novo sobre uma entidade que você quer ensinar.
    # Não é uma adivinhação: toda resposta "sim"/"não" é gravada na hora como um fato
    # novo na base (observado), concatenando conhecimento real sobre o domínio. A única
    # parte "esperta" é a ORDEM das perguntas: a cada rodada, a IA escolhe o atributo que
    # mais ajuda a situar a entidade nova entre as CLASSES que ela já conhece (mesmo
    # cálculo de ganho de informação da seção 4, aplicado aqui para priorizar a pergunta,
    # não para eliminar candidatos). No fim, ela sugere a classe mais parecida e você
    # confirma. A comparação é feita contra CLASSES (Ave, Mamifero, Peixe...), não contra
    # entidades individuais: uma classe reúne o conhecimento de todos os seus exemplos
    # (via herança, seção 3.3), então o vocabulário de perguntas fica mais denso e menos
    # sujeito a "buracos" de uma entidade específica não ter um fato registrado.

    def _classes_de_referencia(self, universo=None):
        """Classes realmente usadas: qualquer uma que já seja objeto de um é_um (mesmo
        que também tenha subclasses, ex. 'Ave' continua sendo usada diretamente por
        Pato e Cisne, mesmo existindo a subclasse mais específica 'AveAquatica')."""
        if universo is not None:
            return [self._normalizar_entidade(e) for e in universo]
        classes = {f.o for f in self.base.consultar(r="é_um", minimo=LIMIAR_ACEITE)}
        return sorted(classes)

    def _atributos_akinator(self, entidade):
        ignorar = {"é_um", "subclasse_de"}
        return {(f.r, f.o) for f in self.base.consultar(s=entidade, minimo=LIMIAR_ACEITE)
                if f.r not in ignorar}

    def _profundidade_classe(self, classe, memo=None, visitando=None):
        """1 para uma classe-raiz; +1 a cada subclasse_de até a raiz. Protegido contra
        ciclo em subclasse_de (ex.: criado sem querer via retroalimentação): se uma
        classe aparecer no próprio caminho de busca, para ali em vez de recursar pra
        sempre e estourar RecursionError."""
        memo = {} if memo is None else memo
        if classe in memo:
            return memo[classe]
        visitando = set() if visitando is None else visitando
        if classe in visitando:
            return 1  # ciclo detectado: trata como se não tivesse mais pai
        visitando.add(classe)
        pais = [f.o for f in self.base.consultar(s=classe, r="subclasse_de", minimo=LIMIAR_ACEITE)]
        memo[classe] = 1 + max((self._profundidade_classe(p, memo, visitando) for p in pais), default=0)
        visitando.discard(classe)
        return memo[classe]

    def _descricao_do_dominio(self, classes):
        """Frase de apresentação: 'Ave (Pato, Cisne, Pinguim...); Mamifero (Gato...)'."""
        partes = []
        for classe in sorted(classes):
            membros = sorted(f.s for f in self.base.consultar(r="é_um", o=classe,
                                                               minimo=LIMIAR_ACEITE))
            exemplos = ", ".join(membros[:3]) + ("..." if len(membros) > 3 else "")
            partes.append(f"{classe} ({exemplos})" if membros else classe)
        return "; ".join(partes)

    def entrevista_fluxo(self, entidade=None, universo=None, max_perguntas=10,
                         limiar_parada=0.85):
        """Entrevista a entidade `entidade` (pede o nome se vier None), perguntando os
        atributos mais informativos do domínio e gravando cada resposta como fato novo."""
        if not entidade:
            entidade = (yield ("Sou um sistema especializado neste domínio de "
                              "conhecimento. Qual é o nome da entidade (pessoa, "
                              "animal, item...) que você quer me ensinar?")).strip()
        if not entidade:
            return "Nenhum nome informado. Entrevista cancelada."
        entidade = self._normalizar_entidade(entidade)

        classes = self._classes_de_referencia(universo)
        atributos_de = {c: self._atributos_akinator(c) for c in classes}
        ja_sabidos = self._atributos_akinator(entidade)  # caso a entidade já exista
        universo_perguntas = {(r, o) for atrs in atributos_de.values() for r, o in atrs
                              # pula a "metade negativa" de uma relação com oposto
                              # declarado, já é coberta pela relação positiva, e o
                              # que a entidade já tem registrado.
                              if r not in self.base.negacoes.values() and (r, o) not in ja_sabidos}

        # Prioriza LEVEMENTE classes mais específicas desde o início: "AveAquatica" é
        # uma sugestão mais útil que "Animal" quando as duas se encaixam igualmente bem
        # nas respostas. O peso usa a profundidade na hierarquia, mas suavizado
        # (1 + profundidade), não proporcional puro, senão uma subclasse que herda
        # exatamente os mesmos atributos da classe-mãe (sem nenhum atributo próprio
        # que a distinga, caso de AveAquatica em relação a Ave neste domínio de
        # exemplo) venceria SEMPRE por profundidade, mesmo sem nenhuma evidência real
        # a favor dela especificamente, limitação conhecida quando a subclasse não
        # tem nenhum atributo que a separe da classe-mãe nos dados.
        profundidades = {c: self._profundidade_classe(c) for c in classes}
        pesos = {c: 1 + profundidades[c] for c in classes}
        soma_pesos = sum(pesos.values()) or 1
        parecenca = {c: pesos[c] / soma_pesos for c in classes} if classes else {}
        perguntados = set()
        aprendidos = []
        rodada = 0
        introducao = (f"Sou um sistema especializado neste domínio (conheço: "
                      f"{self._descricao_do_dominio(classes)}). Vou te fazer "
                      f'perguntas para aprender sobre "{entidade}" e encaixá-la no que '
                      "já sei.\n\n") if classes else ""
        while rodada < max_perguntas:
            melhor = None  # (quão pouco a pergunta ajudaria a situar a entidade, atributo)
            # sorted(): em caso de empate, o desempate é sempre o mesmo (ordem alfabética),
            # em vez de depender da ordem interna (aleatória entre execuções) de um set.
            for (r, o) in sorted(universo_perguntas - perguntados):
                oposta = self.base.relacao_oposta(r)
                tem = sum(parecenca[c] for c in classes if (r, o) in atributos_de[c])
                nao_tem = sum(parecenca[c] for c in classes
                              if oposta and (oposta, o) in atributos_de[c])
                desconhecido = max(0.0, 1 - tem - nao_tem)
                # menor = pergunta mais útil: a que melhor divide as classes ainda
                # plausíveis, evitando perguntas sobre as quais quase nenhuma classe
                # tem posição definida.
                pontuacao = abs(tem - nao_tem) + 2 * desconhecido
                if melhor is None or pontuacao < melhor[0]:
                    melhor = (pontuacao, (r, o))
            if melhor is None:
                break
            _, (r, o) = melhor
            perguntados.add((r, o))
            rodada += 1
            frase = f'tem/é "{o}"' if r == "tem_propriedade" else f'{r.replace("_", " ")} "{o}"'
            texto = f'[{rodada}/{max_perguntas}] "{entidade}" {frase}? (sim / não / não sei)'
            resp = (yield (introducao + texto)).strip().lower()
            introducao = ""
            oposta = self.base.relacao_oposta(r)
            if resp in ("sim", "s"):
                self.base.adicionar(entidade, r, o, 1.0, "observado")
                aprendidos.append((r, o))
                certo, errado = (r, o), (oposta, o) if oposta else None
            elif resp in ("não", "nao", "n"):
                if oposta:
                    self.base.adicionar(entidade, oposta, o, 1.0, "observado")
                    aprendidos.append((oposta, o))
                certo, errado = (oposta, o) if oposta else None, (r, o)
            else:
                certo = errado = None  # "não sei": não atualiza a parecença, segue perguntando
            if certo is not None or errado is not None:
                for c in classes:
                    if certo is not None and certo in atributos_de[c]:
                        parecenca[c] *= 0.85
                    elif errado is not None and errado in atributos_de[c]:
                        # Antes a penalidade (0,05) era muito mais forte que o reforço
                        # (0,95) em relação ao neutro (0,5): uma única contradição
                        # pesava mais do que várias perguntas "não sei" seguidas, porque
                        # propriedades de classe são DEFAULTS com exceção (seção 4.2),
                        # não regras rígidas, um avestruz contrariar "Ave tem_propriedade
                        # voa" não deveria valer mais que isso. Calibrado agora pra ficar
                        # aproximadamente simétrico em log-odds com o reforço (0,35 fica
                        # tão longe do neutro 0,55 para baixo quanto 0,85 fica para cima).
                        parecenca[c] *= 0.35
                    else:
                        parecenca[c] *= 0.55  # a classe não define esse atributo
                soma = sum(parecenca.values()) or 1.0
                parecenca = {c: v / soma for c, v in parecenca.items()}
                if parecenca and max(parecenca.values()) >= limiar_parada:
                    break

        self.pensar(verbose=False)
        linhas = [f'Registrei {len(aprendidos)} fato(s) novo(s) sobre "{entidade}"' +
                 (": " + "; ".join(f"{r} {o}" for r, o in aprendidos) + "."
                  if aprendidos else ", mas nenhuma resposta direta ainda.")]
        classe_sugerida = max(parecenca, key=parecenca.get) if parecenca else None
        if classe_sugerida:
            linhas.append(f'Pelo que você contou, "{entidade}" parece pertencer à '
                          f'classe "{classe_sugerida}" ({parecenca[classe_sugerida]:.0%} '
                          "de confiança).")
        pergunta_classe = (yield ("\n".join(linhas) +
                                  f'\n\nA que classe "{entidade}" pertence de verdade? '
                                  f"(ex.: {classe_sugerida or 'Ave'}; "
                                  "digite 'pular' para não classificar agora)")).strip()
        if pergunta_classe and pergunta_classe.lower() not in ("pular", "não sei", "nao sei"):
            classe = self._normalizar_entidade(pergunta_classe)
            self.base.adicionar(entidade, "é_um", classe, 1.0, "observado")
            self.pensar(verbose=False)
            return (f'Pronto: "{entidade}" agora faz parte da base como {classe}, com '
                    f"{len(aprendidos) + 1} fato(s) novo(s) registrados. Quanto mais "
                    "entidades você ensinar, mais afiadas ficam as próximas entrevistas.")
        return (f'Tudo bem: guardei {len(aprendidos)} fato(s) sobre "{entidade}" sem '
                f"classe definida. Você pode classificar depois com 'classifique {entidade}'.")

    def entrevistar(self, entidade=None, universo=None, max_perguntas=12, limiar_parada=0.85):
        """Versão por linha de comando (input()) da entrevista de conhecimento."""
        gen = self.entrevista_fluxo(entidade, universo, max_perguntas, limiar_parada)
        return self._rodar_gerador_com_input(gen, lambda p: f"  [IA] {p} ")

    def perguntar(self, texto, interativo=False):
        print(f">> {texto}")
        print(self.linguagem.processar(texto, interativo=interativo))
        print()

    def explicar(self, s, r, o):
        print("\n".join(self.motor.explicar(s, r, o)))

    def o_que_falta(self, entidade, imprimir=True):
        itens = self.motor.abduzir(s=entidade)
        if not itens:
            msg = f"Nenhuma regra está parcialmente satisfeita para {entidade}."
        else:
            msg = "\n".join(f"  Falta {fmt(falt)} para concluir {fmt(conc)} "
                            f"(regra '{nome}', conf {conf:.2f})" for conf, nome, falt, conc in itens)
            msg = f"Preciso de mais fatos sobre {entidade}:\n" + msg
        if imprimir:
            print(msg)
        return msg

    def treinar_classificador(self, classes):
        self.classificador.treinar(classes)
        print(f"Classificador treinado com {sum(len(v) for v in self.classificador.exemplos.values())} "
              f"exemplos e {len(self.classificador.vocab)} atributos.")

    def classificar(self, entidade, adicionar=True, limiar=0.75, imprimir=True):
        if not self.classificador.classes:
            # Ainda não foi treinado (ex.: logo depois de SistemaIA.carregar(), ou se o
            # usuário nunca chamou treinar_classificador manualmente), treina agora,
            # usando as classes já cadastradas, em vez de quebrar com AttributeError.
            self.treinar_classificador(self._classes_de_referencia())
        probs = self.classificador.classificar(entidade)
        linhas = [f"Classificação de {entidade} (atributos: {sorted(self.classificador.atributos(entidade))})"]
        linhas += [f"  {c:<10} {p:6.1%}" for c, p in probs]
        classe, p = probs[0]
        if adicionar and p >= limiar:
            self.base.adicionar(entidade, "é_um", classe, p, "classificador")
            self.pensar(verbose=False)
            linhas.append(f"  Hipótese aceita: {entidade} é_um {classe} (conf {p:.2f})")
        elif adicionar:
            linhas.append(f"  Confiança abaixo de {limiar:.0%}. Preciso de mais fatos.")
        msg = "\n".join(linhas)
        if imprimir:
            print(msg)
        return msg

    def treinar_embeddings(self, **kw):
        self.embeddings = EmbeddingsTransE(**kw).treinar(self.base)
        print(f"TransE treinado: {len(self.embeddings.ents)} entidades, "
              f"{len(self.embeddings.rels)} relações, perda final {self.embeddings.perdas[-1]:.2f}")

    # Relatórios e visualização
    def relatorio(self):
        cont = Counter(f.fonte for f in self.base.fatos.values())
        print(f"Fatos: {len(self.base.fatos)} {dict(cont)} | rejeitados: {len(self.base.rejeitados)}")
        print(f"Regras: {len(self.motor.regras)}")
        for r in sorted(self.motor.regras, key=lambda x: (x.origem, x.nome)):
            print(f"  {r}  disparos={r.disparos}")

    def listar(self, fonte=None, relacao=None):
        for f in sorted(self.base.fatos.values(), key=lambda x: x.chave):
            if (fonte is None or f.fonte == fonte) and (relacao is None or f.r == relacao):
                print(f"  {f}")

    CORES = {"observado": "#2b6cb0", "confirmado": "#2f855a", "inferido": "#dd6b20",
             "classificador": "#805ad5", "hipotese": "#c53030"}

    def visualizar(self, entidades=None, titulo="Grafo de conhecimento", relacoes_ocultas=()):
        G = nx.DiGraph()
        for f in self.base.fatos.values():
            if f.confianca < LIMIAR_ACEITE or f.r in relacoes_ocultas:
                continue
            if entidades is not None and not (f.s in entidades and f.o in entidades):
                continue
            if G.has_edge(f.s, f.o):
                G[f.s][f.o]["rotulo"] += f"\n{f.r}"
            else:
                G.add_edge(f.s, f.o, rotulo=f.r, fonte=f.fonte)
        plt.figure(figsize=(16, 11))
        pos = nx.spring_layout(G, seed=7, k=1.6 / math.sqrt(max(len(G), 1)), iterations=200)
        nx.draw_networkx_nodes(G, pos, node_color="#edf2f7", edgecolors="#4a5568", node_size=1900)
        nx.draw_networkx_labels(G, pos, font_size=8)
        for fonte, cor in self.CORES.items():
            arestas = [(u, v) for u, v, d in G.edges(data=True) if d["fonte"] == fonte]
            if arestas:
                nx.draw_networkx_edges(G, pos, edgelist=arestas, edge_color=cor, arrows=True,
                                       arrowsize=14, node_size=1900, width=1.4,
                                       style="solid" if fonte in FONTES_FIRMES else "dashed",
                                       connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_edge_labels(G, pos, font_size=6,
                                     edge_labels={(u, v): d["rotulo"] for u, v, d in G.edges(data=True)})
        plt.legend(handles=[Line2D([0], [0], color=c, lw=2, label=f) for f, c in self.CORES.items()],
                   loc="upper left")
        plt.title(titulo)
        plt.axis("off")
        plt.show()

    def visualizar_embeddings(self):
        emb = self.embeddings
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
        a1.plot(emb.perdas)
        a1.set_title("Perda do TransE por época")
        a1.set_xlabel("época")
        X = emb.E - emb.E.mean(axis=0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        P = U[:, :2] * S[:2]
        a2.scatter(P[:, 0], P[:, 1], s=20)
        for (x, y), nome in zip(P, emb.ents):
            a2.annotate(nome, (x, y), fontsize=7)
        a2.set_title("Entidades projetadas em 2D (PCA dos embeddings)")
        plt.show()

    # Persistência
    def salvar(self, caminho="memoria_ia.json"):
        b = self.base
        dados = {
            "fatos": [{"s": f.s, "r": f.r, "o": f.o, "alpha": f.alpha, "beta": f.beta,
                       "fonte": f.fonte, "proc": f.procedencias, "valido_ate": f.valido_ate,
                       "justificativa": [f.justificativa[0], [list(k) for k in f.justificativa[1]]]
                       if f.justificativa else None}
                      for f in b.fatos.values()],
            "regras": [{"nome": r.nome, "premissas": r.premissas, "conclusao": r.conclusao,
                        "alpha": r.alpha, "beta": r.beta, "origem": r.origem, "metricas": r.metricas}
                       for r in self.motor.regras if r.origem != "estrutural"],
            "rejeitados": [list(k) for k in b.rejeitados],
            "ontologia": {"transitivas": sorted(b.transitivas), "simetricas": sorted(b.simetricas),
                          "inversas": b.inversas, "funcionais": sorted(b.funcionais),
                          "negacoes": b.negacoes, "disjuntas": b.disjuntas},
            "contador_inducao": self.indutor.contador,
            "formas": b.formas, "vocabulario": b.vocabulario, "config": b.config,
            "exibicao": dict(EXIBICAO), "faq": [list(x) for x in b.faq],
            "rotulos": [list(x) for x in b.rotulos], "fontes_stats": b.fontes_stats,
            "penalizados": [[list(a), list(c)] for a, c in b.penalizados],
        }
        with open(caminho, "w", encoding="utf-8") as fp:
            json.dump(dados, fp, ensure_ascii=False, indent=1)
        print(f"Memória salva em {caminho}: {len(dados['fatos'])} fatos, {len(dados['regras'])} regras.")

    @classmethod
    def carregar(cls, caminho="memoria_ia.json"):
        with open(caminho, encoding="utf-8") as fp:
            dados = json.load(fp)
        ia = cls()
        b = ia.base
        ont = dados["ontologia"]
        b.transitivas = set(ont["transitivas"])
        b.simetricas = set(ont["simetricas"])
        b.inversas = dict(ont["inversas"])
        b.funcionais = set(ont["funcionais"])
        b.negacoes = dict(ont["negacoes"])
        b.disjuntas = [tuple(x) for x in ont["disjuntas"]]
        for d in dados["regras"]:
            nova = Regra(d["nome"], d["premissas"], d["conclusao"], origem=d["origem"],
                         alpha=d["alpha"], beta=d["beta"], metricas=d["metricas"])
            armazenada = ia.motor.adicionar_regra(nova)
            armazenada.alpha, armazenada.beta = d["alpha"], d["beta"]
        for d in dados["fatos"]:
            just = (d["justificativa"][0], [tuple(k) for k in d["justificativa"][1]]) \
                if d["justificativa"] else None
            f = b.adicionar(d["s"], d["r"], d["o"], 0.5, d["fonte"], just)
            f.alpha, f.beta = d["alpha"], d["beta"]
            f.procedencias = d.get("proc") or []
            f.valido_ate = d.get("valido_ate")
        b.rejeitados = {tuple(k) for k in dados["rejeitados"]}
        b.formas = dict(dados.get("formas", {}))
        b.config = dict(dados.get("config", {}))
        b.vocabulario = dict(dados.get("vocabulario", {}))
        EXIBICAO.update(dados.get("exibicao", {}))
        b.faq = [tuple(x) for x in dados.get("faq", []) if len(x) == 3]
        b.rotulos = [tuple(x) for x in dados.get("rotulos", [])]
        b.fontes_stats = {k: list(v) for k, v in dados.get("fontes_stats", {}).items()}
        b.penalizados = {(tuple(a), tuple(c)) for a, c in dados.get("penalizados", [])}
        ia.indutor.contador = dados.get("contador_inducao", 0)
        ia.pensar(verbose=False)
        print(f"Memória carregada de {caminho}: {len(b.fatos)} fatos, {len(ia.motor.regras)} regras.")
        return ia


# %% [7] DEMO 1: criação da IA, ontologia e observação dos fatos
# Assunto único e fixo: ZOOLOGIA. Toda a demonstração a partir daqui gira em torno de
# animais, taxonomia (quem é ave, mamífero, peixe), relações ecológicas (quem caça
# quem, quem compete por comida, quem é parceiro de quem) e genealogia evolutiva. Isso
# evita misturar assuntos diferentes na mesma base (o que, como vimos na seção 6.2 da
# documentação, prejudicava o modo entrevista). Se você quiser adaptar para outro
# assunto, troque só este bloco: o resto do notebook não presume nada sobre zoologia
# especificamente, só sobre a FORMA dos fatos (sujeito, relação, objeto).

ia = SistemaIA()

# Ontologia das relações do domínio
ia.base.declarar_inversa("predador_de", "presa_de")
ia.base.declarar_simetrica("parceiro_de")
ia.base.declarar_transitiva("evoluiu_de")
ia.base.declarar_funcional("habitat_principal")
ia.base.declarar_disjuntas("Ave", "Mamifero")
ia.base.declarar_disjuntas("Mamifero", "Peixe")

# Regra do CÓDIGO 1, agora como regra de conhecimento com grau de confiança
ia.motor.adicionar_regra(Regra(
    "ave_aquatica", [("?a", "tem_propriedade", "penas"), ("?a", "tem_propriedade", "nada")],
    ("?a", "é_um", "AveAquatica"), 0.85, "manual"))

# Triplas do CÓDIGO 2 (originais), adaptadas para o domínio único: o aluno/disciplina
# virou predador/presa, mas a estrutura sujeito-relação-objeto é a mesma.
triplas_originais = [
    ("Aguia", "é_um", "Ave"),
    ("Salmao", "é_um", "Peixe"),
    ("Aguia", "predador_de", "Salmao"),
    ("Tubarao", "é_um", "Peixe"),
    ("Tubarao", "predador_de", "Salmao"),
]
ia.observar(triplas_originais)

# Taxonomia: hierarquia de classes e suas propriedades herdadas (igual ao código 1,
# generalizado para toda a árvore de animais).
ia.observar([
    ("Ave", "subclasse_de", "Animal"), ("Mamifero", "subclasse_de", "Animal"),
    ("Peixe", "subclasse_de", "Animal"), ("Reptil", "subclasse_de", "Animal"),
    ("Animal", "subclasse_de", "SerVivo"),
    ("AveAquatica", "subclasse_de", "Ave"),
    ("Animal", "tem_propriedade", "respira"),
    ("Ave", "tem_propriedade", "penas"), ("Ave", "tem_propriedade", "voa"),
    ("Mamifero", "tem_propriedade", "pelos"), ("Peixe", "tem_propriedade", "nada"),
    ("Peixe", "tem_propriedade", "escamas"), ("Reptil", "tem_propriedade", "escamas"),
    ("Pato", "é_um", "Ave"), ("Pato", "tem_propriedade", "nada"),
    ("Pinguim", "é_um", "Ave"), ("Pinguim", "tem_propriedade", "nada"),
    ("Pinguim", "não_tem_propriedade", "voa"),  # exceção à regra geral
    ("Aguia", "tem_propriedade", "penas"),
    ("Cisne", "é_um", "Ave"), ("Cisne", "tem_propriedade", "nada"),
    ("Gato", "é_um", "Mamifero"), ("Cachorro", "é_um", "Mamifero"),
    ("Lobo", "é_um", "Mamifero"), ("Golfinho", "é_um", "Mamifero"),
    ("Golfinho", "tem_propriedade", "nada"),
    ("Salmao", "tem_propriedade", "escamas"), ("Tubarao", "tem_propriedade", "escamas"),
    ("Cobra", "é_um", "Reptil"), ("Cobra", "não_tem_propriedade", "pelos"),
])

# Relações ecológicas: cadeia alimentar (predador_de/presa_de, inversa), genealogia
# evolutiva (evoluiu_de, transitiva), parceria reprodutiva (parceiro_de, simétrica) e
# habitat (funcional, um animal "mora", principalmente, num só lugar).
ia.observar([
    ("Golfinho", "predador_de", "Salmao"),
    ("Gato", "predador_de", "Rato"), ("Cobra", "predador_de", "Sapo"),
    # exemplos parciais de "compete_com" entre os 3 predadores de Salmao (Aguia,
    # Tubarao, Golfinho): faltando só o par Aguia/Golfinho de propósito, pra IA
    # descobrir sozinha (seção 9) que dois predadores da mesma presa competem.
    ("Aguia", "compete_com", "Tubarao"), ("Tubarao", "compete_com", "Aguia"),
    ("Tubarao", "compete_com", "Golfinho"), ("Golfinho", "compete_com", "Tubarao"),
    ("Cachorro", "evoluiu_de", "Lobo"), ("Lobo", "evoluiu_de", "Canideo"),
    ("Canideo", "evoluiu_de", "Carnivoro"),
    ("Pato", "parceiro_de", "Pata"), ("Pata", "é_um", "Ave"),
    ("Golfinho", "habitat_principal", "Oceano"),
    ("Golfinho", "habitat_principal", "RioDoce", 0.6),  # conflita com o fato acima, e é menos confiável
    ("Salmao", "habitat_principal", "Rio"), ("Tubarao", "habitat_principal", "Oceano"),
])

# O dicionário original do CÓDIGO 1 entra como uma entidade desconhecida
animal = {"tem_penas": True, "nada": True, "voa": False}
ia.observar(SistemaIA.dicionario_para_triplas("Misterio", animal))
ia.observar([("Criatura", "tem_propriedade", "nada")])

# Impressão no formato do CÓDIGO 2
for sujeito, relacao, objeto in triplas_originais:
    print(f"{sujeito} --{BaseConhecimento.normalizar_relacao(relacao)}--> {objeto}")
print(f"\nFatos observados: {len(ia.base.fatos)}")


# %% [8] DEMO 2: raciocínio dedutivo com exceções e conflitos

ia.pensar()
print("\nAlgumas conclusões inferidas:")
for f in ia.base.consultar(minimo=LIMIAR_ACEITE):
    if f.fonte == "inferido" and f.s in ("Pinguim", "Misterio", "Cachorro", "Pata", "Salmao"):
        print(f"  {f}")

print("\nExplicação de uma conclusão:")
ia.explicar("Misterio", "é_um", "Animal")


# %% [9] DEMO 3: aprendizado de regras a partir dos dados (indução)

novas = ia.aprender_regras()
print("\nNovos 'compete_com' descobertos:")
ia.listar(fonte="inferido", relacao="compete_com")
print()
ia.explicar("Aguia", "compete_com", "Golfinho")


# %% [10] DEMO 4: aprendizado por feedback (a IA corrige as próprias regras)

print("Antes:")
for r in novas:
    print(f"  {r}")
print()
ia.feedback("Aguia", "compete_com", "Golfinho", correto=True)
ia.feedback("Golfinho", "compete_com", "Aguia", correto=True)
print("\nDepois:")
for r in novas:
    print(f"  {r}")


# %% [11] DEMO 5: código 1 evoluído (classificação e "preciso de mais fatos")

ia.treinar_classificador(["Ave", "Mamifero", "Peixe"])
print()
ia.classificar("Misterio")
print()
ia.classificar("Criatura")
print()
ia.o_que_falta("Criatura")
print()
print("Conclusão sobre Misterio (equivalente ao print do código 1):")
print("  pode ser uma ave aquática" if ia.base.fatos.get(("Misterio", "é_um", "AveAquatica"))
      else "  preciso de mais fatos")
ia.explicar("Misterio", "é_um", "AveAquatica")


# %% [12] DEMO 6: intuição estatística com embeddings (TransE)

ia.treinar_embeddings(dim=32, epocas=300)
print("\nFatos ausentes mais plausíveis segundo os embeddings:")
for p, d, t in ia.embeddings.sugerir(ia.base, ["predador_de", "compete_com", "evoluiu_de"], top_k=8):
    print(f"  {fmt(t):<45} plausibilidade={p:.2f}  distância={d:.2f}")
ia.visualizar_embeddings()


# %% [13] DEMO 7: perguntas em linguagem natural

for pergunta in [
    "quem predador_de Salmao?",
    "o que Golfinho predador_de?",
    "Golfinho é um Mamifero?",
    "Pinguim tem_propriedade voa?",
    "Pato é uma AveAquatica?",
    "Gato compete_com Cobra?",
    "por que Cachorro evoluiu_de Carnivoro?",
    "Cobra predador_de Rato",
    "Gato compete_com Cobra?",
    "o que falta para Criatura?",
]:
    ia.perguntar(pergunta)


# %% [14] DEMO 7b: retroalimentação, a IA pergunta de volta até entender
# Simula o que aconteceria no prompt (célula 20): quando a confiança não basta,
# a IA usa a abdução para achar a premissa que falta e PERGUNTA, repetindo até
# concluir. Aqui as respostas do usuário são simuladas (builtins.input trocado
# por uma fila de respostas), só para a demonstração rodar sem pausar.

import builtins

def _simular_respostas(respostas):
    fila = list(respostas)

    def _input_falso(prompt=""):
        print(prompt, end="")
        resp = fila.pop(0) if fila else "não sei"
        print(resp)
        return resp
    return _input_falso

print("Pergunta sem informação suficiente, com retroalimentação ligada:\n")
print("(antes de perguntar, a IA não tem como saber se Cachorro compete com Gato:")
print(" Gato predador_de Rato é conhecido, mas Cachorro predador_de Rato não está na base)\n")
original_input = builtins.input
builtins.input = _simular_respostas(["sim"])
try:
    # A IA não sabe se Cachorro compete com Gato. Vai abduzir que, se Cachorro também
    # fosse predador de Rato (que Gato já é), concluiria pela regra induzida na seção 9.
    # Pergunta, recebe "sim", aprende o fato, reinfere e responde.
    resposta = ia.resolver_fato("Cachorro", "compete_com", "Gato", interativo=True)
finally:
    builtins.input = original_input
print("\nResposta final:", resposta)

print("\n" + "-" * 70)
print("Pergunta 'quem', sem resultado, com retroalimentação ligada:\n")
print("(ninguém foi cadastrado como predador de Golfinho ainda)\n")
builtins.input = _simular_respostas(["Tubarao"])
try:
    resposta = ia.resolver_quem("predador_de", "Golfinho", interativo=True)
finally:
    builtins.input = original_input
print("\nResposta final:", resposta)
print("(repare que 'Aguia predador_de Golfinho' também apareceu sozinho: a regra")
print(" induzida_4, uma das regras 'ruins' da seção 9/10, concluiu isso a partir de")
print(" 'Tubarao compete_com Aguia' + o fato que você acabou de ensinar, exatamente")
print(" o tipo de dedução estranha que o feedback da seção 5.2 serve para corrigir.)")

print("\n" + "-" * 70)
print("Usuário também não sabe (a IA desiste de forma elegante):\n")
print("(Golfinho e Pinguim têm só UMA pista em comum na base, a IA pergunta uma vez,")
print(" recebe 'não sei', e não sobra mais nenhuma pista pra tentar)\n")
builtins.input = _simular_respostas(["não sei", "não sei", "não sei", "não sei"])
try:
    resposta = ia.resolver_fato("Golfinho", "compete_com", "Pinguim", interativo=True)
finally:
    builtins.input = original_input
print("\nResposta final:", resposta)

print("\nPara usar de verdade (você respondendo de verdade), use o chat da célula\n"
      "'Conversa interativa', no fim do notebook: qualquer pergunta com '?' que a\n"
      "IA não souber responder direto vai gerar perguntas de volta, até ela\n"
      "entender ou você dizer que também não sabe.")


# %% [15] DEMO 7c: modo entrevista, a IA se apresenta como especialista e aprende
# conhecimento novo sobre uma entidade que você traz. Diferente da retroalimentação da
# seção 6 (você pergunta, a IA completa), aqui é a IA que pergunta, mas o objetivo NÃO
# é adivinhar: cada "sim"/"não" vira fato novo na base na hora. Para a demo rodar sem
# pausar, um "especialista simulado" responde como alguém que conhece de verdade um
# avestruz (entidade nova, que ainda não existe na base) responderia.

def _especialista_simulado(fatos_reais, classe_real):
    """Responde à entrevista com um conjunto de fatos reais predefinido (simula uma
    pessoa respondendo de verdade sobre uma entidade nova, não cadastrada ainda)."""
    def responder(pergunta):
        m_rodada = re.search(r"\[\d+/\d+\].*", pergunta, re.DOTALL)
        if m_rodada:  # é uma pergunta de atributo, não a de classe
            linha = m_rodada.group(0)  # descarta a introdução, que também tem aspas
            aspas = re.findall(r'"([^"]+)"', linha)
            valor = aspas[-1]  # a 1ª aspa é sempre o nome da entidade; a última é o atributo
            resto = linha.split(f'"{aspas[0]}"', 1)[1]  # tudo depois do nome da entidade
            relacao = ("tem_propriedade" if 'tem/é "' in resto
                      else re.search(r'^\s*(.+?)\s*"', resto).group(1).replace(" ", "_"))
            if (relacao, valor) in fatos_reais:
                return "sim"
            if (ia.base.relacao_oposta(relacao), valor) in fatos_reais:
                return "não"
            return "não sei"
        if "classe" in pergunta.lower() and "pertence" in pergunta.lower():
            return classe_real
        return "não sei"
    return responder

# Fatos reais de um avestruz, do jeito que uma pessoa responderia (penas sim, mas não
# voa e não nada, diferente de todas as aves já cadastradas na base de demonstração).
fatos_avestruz = {("tem_propriedade", "penas"), ("tem_propriedade", "respira"),
                  ("não_tem_propriedade", "voa"), ("não_tem_propriedade", "nada"),
                  ("não_tem_propriedade", "escamas"), ("não_tem_propriedade", "pelos")}

print(f"Antes da entrevista, 'Avestruz' existe na base? "
      f"{'Avestruz' in ia.base.entidades()}\n")

responder = _especialista_simulado(fatos_avestruz, "Ave")
gen = ia.entrevista_fluxo("Avestruz")
pergunta = next(gen)
try:
    while True:
        resp = responder(pergunta)
        print(pergunta, "\n  ->", resp, "\n")
        pergunta = gen.send(resp)
except StopIteration as fim:
    print("RESULTADO FINAL:", fim.value)

print(f"\nDepois da entrevista, fatos novos sobre Avestruz:")
for f in ia.base.consultar(s="Avestruz"):
    print(" ", f)

print("\nNote que a entrevista NÃO perguntou sobre tudo que existe na base, ela\n"
      "priorizou as propriedades físicas mais úteis para situar o Avestruz entre o que\n"
      "já se conhecia (penas, voa, nada...), ignorando relações ecológicas como\n"
      "'compete_com' ou 'evoluiu_de', que nenhuma classe define diretamente (só\n"
      "entidades individuais têm esses fatos, não Ave/Mamifero/Peixe em si).\n"
      "\n"
      "Para conversar de verdade, use o prompt (célula 20): ele carrega uma base de\n"
      "arquivo e passa a fazer essas perguntas por iniciativa própria.")


# %% [16] DEMO 8: visualização do grafo

taxonomia = {"Pato", "Pinguim", "Cisne", "Aguia", "Misterio", "Criatura", "Golfinho", "Tubarao",
            "Salmao", "Gato", "Cachorro", "Cobra", "Ave", "AveAquatica", "Mamifero", "Peixe",
            "Reptil", "Animal", "penas", "nada", "voa"}
ia.visualizar(taxonomia, "Taxonomia: quem é ave, mamífero, peixe (observado x inferido)",
             relacoes_ocultas=("subclasse_de",))
ecologia = {"Aguia", "Tubarao", "Golfinho", "Salmao", "Gato", "Cobra", "Rato", "Cachorro",
           "Lobo", "Canideo", "Carnivoro", "Pato", "Pata"}
ia.visualizar(ecologia, "Relações ecológicas: predador_de, compete_com, evoluiu_de, parceiro_de")


# %% [17] DEMO 9: relatório e memória persistente

ia.relatorio()
ia.salvar("memoria_ia.json")
ia2 = SistemaIA.carregar("memoria_ia.json")
print(ia2.verificar("Cachorro", "compete_com", "Gato"))


# %% [18] Chat: marca, limites e utilidades de texto
# Interface sem menus e sem botões: a pessoa escreve frases e a IA responde em texto,
# como num terminal. A IA conduz a conversa: pergunta o que falta, confronta o que
# contradiz a base e confirma as próprias deduções. Há três camadas:
#   Interprete : transforma uma frase em português em uma intenção (afirmar, perguntar...)
#   Dialogo    : mantém o estado da conversa e decide a próxima pergunta da IA
#   Prompt     : a aparência de terminal (ipywidgets) e prompt_terminal() (input())
# Subdiálogos são geradores: cada yield devolve (texto, espera) e recebe a resposta da
# pessoa. espera = "sim_nao" (a resposta chega como "sim", "nao" ou "nao_sei") ou "livre".

import os
import io
import csv
import time
import inspect
import base64
import difflib
import textwrap
import zipfile
import urllib.parse
import urllib.request
import html as _html
import unicodedata
from datetime import datetime
from contextlib import redirect_stdout
from functools import lru_cache

# Identidade visual. Azul e laranja medidos no portal da Datainfo (datainfo.inf.br), não de manual de
# marca; o tom escuro é derivado do azul. Troque pelos oficiais se a empresa fornecer outros.
MARCA = {
    "empresa": "DataInfo",
    "ia": "Info",
    "acento_claro": "#114984",
    "acento_escuro": "#6fa3ef",
    "acento_destaque": "#ff7a00",
    "logo_svg": None,   # caminho de um arquivo .svg com o logotipo, opcional
}


def _carregar_marca(caminho="marca.json"):
    """Sobrescreve MARCA com o conteúdo de marca.json, se o arquivo existir ao lado do notebook.
    Chaves aceitas: empresa, ia, acento_claro, acento_escuro, acento_destaque (#rrggbb) e logo_svg (caminho)."""
    try:
        with open(caminho, encoding="utf-8") as fp:
            dados = json.load(fp)
    except (OSError, ValueError):
        return False
    for chave in ("empresa", "ia", "logo_svg"):
        if isinstance(dados.get(chave), str) and dados[chave].strip():
            MARCA[chave] = dados[chave].strip()
    for chave in ("acento_claro", "acento_escuro", "acento_destaque"):
        if isinstance(dados.get(chave), str) and re.match(r"^#[0-9a-fA-F]{6}$", dados[chave]):
            MARCA[chave] = dados[chave]
    return True


_carregar_marca()

# Limites pensados para o Colab (2 vCPU, cerca de 12 GB de RAM, sem GPU e disco temporário)
LIMITE_INDUCAO = 4000        # fatos informados; acima disso a busca de padrões fica lenta
LIMITE_EMBEDDINGS = 3000     # fatos informados; acima disso o treino de vetores não compensa
LIMITE_DOWNLOAD = 20 * 1024 * 1024
LIMITE_LINK = 8 * 1024 * 1024  # tamanho máximo de arquivo embutido como link de download


def _sem_acento(texto):
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn").lower()


@lru_cache(maxsize=8192)
def _chave(texto):
    return re.sub(r"[\s_\-]+", "", _sem_acento(texto))


def _sem_marcas(texto):
    """Tira só os acentos, preservando maiúsculas e minúsculas."""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _legivel(nome):
    return EXIBICAO.get(nome, nome).replace("_", " ")


def _lista(valor):
    """Normaliza o retorno de um subdiálogo para uma lista de linhas."""
    if valor is None:
        return []
    if isinstance(valor, str):
        return valor.split("\n")
    saida = []
    for item in valor:
        saida.extend(str(item).split("\n"))
    return saida


# %% [19] Chat: interpretação de frases em português
ARTIGOS = {"o", "a", "os", "as", "um", "uma", "uns", "umas"}
NEGACOES = {"nao", "nunca", "jamais"}
FRACOS = {("e",)}
VERBOS_PROPRIEDADE = {"voa", "nada", "respira", "corre", "anda", "canta", "late", "mia",
                      "mama", "rasteja", "amamenta", "poe_ovos", "respira_por_pulmoes",
                      "respira_por_branquias"}

# palavras de nomes de propriedade que a pessoa escreve com acento; só para mostrar na tela
_ACENTOS_PROPRIEDADE = {
    "poe": "põe", "pulmoes": "pulmões", "branquias": "brânquias", "ecolocalizacao": "ecolocalização",
    "carapaca": "carapaça", "superficie": "superfície", "solida": "sólida", "rotacao": "rotação",
    "orbita": "órbita", "agua": "água", "liquida": "líquida", "aneis": "anéis",
    "subterraneo": "subterrâneo", "retrograda": "retrógrada", "numero": "número", "serie": "série",
    "patrimonio": "patrimônio", "atmosfera": "atmosfera", "vulcanismo": "vulcanismo",
    "esqueleto": "esqueleto", "cartilaginoso": "cartilaginoso", "satelites": "satélites",
    "vertebral": "vertebral", "heterotrofo": "heterótrofo", "peconhenta": "peçonhenta"}


def _acentuar_propriedade(nome):
    return " ".join(_ACENTOS_PROPRIEDADE.get(p, p) for p in nome.split(" "))

# primeira forma de cada lista = como a IA escreve a relação
FORMAS_RELACAO = {
    "predador_de": ["caça", "caçam", "preda", "predam", "come", "comem", "devora",
                    "devoram", "se alimenta de", "se alimentam de", "é predador de",
                    "é predadora de"],
    "presa_de": ["é presa de", "é caçado por", "é caçada por", "é comido por",
                 "é comida por"],
    "tem_propriedade": ["tem", "têm", "possui", "possuem"],
    "habitat_principal": ["vive em", "vive no", "vive na", "vivem em", "mora em",
                          "mora no", "mora na", "habita", "habitam"],
    "evoluiu_de": ["evoluiu de", "evoluiu do", "evoluiu da", "descende de",
                   "descende do", "descende da"],
    "compete_com": ["compete com", "disputa comida com", "disputa com"],
    "parceiro_de": ["é parceiro de", "é parceira de", "acasala com"],
    "subclasse_de": ["é um tipo de", "é uma espécie de", "é um subtipo de"],
    "é_um": ["é da classe", "é um", "é uma", "são", "são um", "são uma", "é",
             "pertence à classe"],
}
SUPERFICIE = {"predador_de": "caça", "presa_de": "é presa de", "tem_propriedade": "tem",
              "habitat_principal": "vive em", "evoluiu_de": "evoluiu de",
              "compete_com": "compete com", "parceiro_de": "é parceiro de",
              "subclasse_de": "é um tipo de", "é_um": "é da classe"}
PERGUNTA_VALOR = {"habitat_principal": "Onde {e} vive?"}

HEDGES = [("eu", "acho", "que"), ("acho", "que"), ("eu", "creio", "que"),
          ("creio", "que"), ("eu", "acredito", "que"), ("acredito", "que"),
          ("talvez",), ("parece", "que"), ("pelo", "que", "eu", "sei"),
          ("pelo", "que", "sei")]
ENCHIMENTO = [("olha",), ("entao",), ("bom",), ("sabe", "que"), ("sabia", "que"),
              ("na", "verdade"), ("saiba", "que"), ("veja", "que"), ("lembre", "que"),
              ("eu", "sei", "que"), ("sei", "que"), ("corrija",), ("correcao",), ("corrigindo",)]

_SIM = {"sim", "s", "isso", "correto", "certo", "verdade", "confirmo", "claro", "exato",
        "exatamente", "procede", "uhum", "ok", "pode", "tenho certeza", "com certeza",
        "isso mesmo", "e verdade", "e isso", "sim claro", "sim tenho certeza"}
_NAO = {"nao", "n", "errado", "falso", "mentira", "negativo", "nunca", "nao e",
        "nao e verdade", "nao procede", "nada disso"}
_NAO_SEI = ("nao sei", "nsei", "nao tenho certeza", "sei la", "nao lembro", "talvez",
            "depois", "pular", "passa", "proxima", "nenhuma ideia", "nao faco ideia")


GRAU_PROVAVEL = 0.75
_GRAD_NAO = ("provavelmente nao", "acho que nao", "creio que nao", "talvez nao",
             "quase certeza que nao", "deve ser que nao")
_GRAD_SIM = ("provavelmente sim", "acho que sim", "creio que sim", "talvez sim",
             "quase certeza que sim", "provavelmente", "acho que e", "deve ser que sim")


def _normalizar_resposta(texto):
    return re.sub(r"\s+", " ", re.sub(r"[,;:.!?]", " ", _sem_acento(texto))).strip()


def grau_resposta(texto):
    """1,0 para sim ou não firmes; GRAU_PROVAVEL para "provavelmente", "acho que" e "talvez sim/não"."""
    t = _normalizar_resposta(texto)
    return GRAU_PROVAVEL if t.startswith(_GRAD_NAO + _GRAD_SIM) else 1.0


def classificar_resposta(texto):
    """Devolve "sim", "nao", "nao_sei" ou "outro" (a pessoa seguiu com outro assunto).
    Respostas graduadas ("provavelmente sim") contam como sim ou não; o peso vem de grau_resposta."""
    t = _normalizar_resposta(texto)
    if not t:
        return "outro"
    if t.startswith(_GRAD_NAO):
        return "nao"
    if t.startswith(_GRAD_SIM):
        return "sim"
    if any(t.startswith(p) for p in _NAO_SEI):
        return "nao_sei"
    primeira = t.split()[0]
    if t in _SIM or (primeira in ("sim", "isso", "correto", "certo", "claro", "exato",
                                  "confirmo", "procede") and len(t.split()) <= 6):
        return "sim"
    if t in _NAO or primeira in ("nao", "errado", "falso", "mentira", "negativo"):
        return "nao"
    return "outro"


def frase_fato(s, r, o):
    """Escreve um fato como frase em português."""
    if r in ("tem_propriedade", "não_tem_propriedade"):
        neg = r.startswith("não")
        nome = _acentuar_propriedade(_legivel(o))
        if o in VERBOS_PROPRIEDADE:
            return f"{_legivel(s)} {'não ' if neg else ''}{nome}"
        return f"{_legivel(s)} {'não tem' if neg else 'tem'} {nome}"
    return f"{_legivel(s)} {SUPERFICIE.get(r, _legivel(r))} {_legivel(o)}"


ROTULO_FONTE = {"observado": "informado", "confirmado": "confirmado",
                "inferido": "deduzido", "classificador": "estimado"}


class Interprete:
    """Converte frases em português em intenções, sem menus nem comandos fixos."""

    def __init__(self, dialogo):
        self.dialogo = dialogo

    @property
    def base(self):
        return self.dialogo.ia.base

    # --- vocabulário que depende da base atual ---
    def lexico(self):
        lex = {}

        def add(superficie, rel):
            lex[tuple(_sem_acento(superficie).split())] = rel
        for rel, formas in FORMAS_RELACAO.items():
            for forma in formas:
                add(forma, rel)
        relacoes = {f.r for f in self.base.fatos.values()} | set(FORMAS_RELACAO)
        for rel in relacoes:
            add(rel, rel)
            add(_legivel(rel), rel)
            add("é " + _legivel(rel), rel)
            leg = _legivel(rel)
            if leg.endswith(" em"):
                for prep in ("na", "no", "nas", "nos"):
                    add(leg[:-2] + prep, rel)
            elif leg.endswith(" de"):
                for prep in ("do", "da", "dos", "das"):
                    add(leg[:-2] + prep, rel)
        for frase, rel in self.base.vocabulario.items():
            add(frase, rel)  # sinônimos ensinados em conversa valem mais que o padrão
        return lex

    def entidades(self):
        return {_chave(e): e for e in self.base.entidades()}

    def propriedades(self):
        props = {_chave(v): v for v in VERBOS_PROPRIEDADE}
        props.update({_chave(f.o): f.o for f in self.base.fatos.values()
                      if f.r in ("tem_propriedade", "não_tem_propriedade")})
        return props

    # --- nomes ---
    @staticmethod
    def tokens(texto):
        return re.findall(r"\w+(?:['\-]\w+)*|,", texto)

    @staticmethod
    def _sem_artigos(palavras):
        palavras = list(palavras)
        while palavras and _sem_acento(palavras[0]) in ARTIGOS:
            palavras.pop(0)
        return palavras

    def nome_entidade(self, palavras):
        palavras = self._sem_artigos(palavras)
        if not palavras:
            return None
        chave = _chave(" ".join(palavras))
        mapa = self.entidades()
        if chave in mapa:
            return mapa[chave]
        for sufixo, troca in (("s", ""), ("es", ""), ("oes", "ao"), ("ais", "al")):
            if chave.endswith(sufixo) and chave[:-len(sufixo)] + troca in mapa:
                return mapa[chave[:-len(sufixo)] + troca]
        for sufixo, troca in (("s", ""), ("es", ""), ("oes", "ao"), ("ais", "al")):
            # sentido contrário: a base tem o plural e a frase trouxe o singular
            if troca and not chave.endswith(troca):
                continue
            candidato = (chave[:-len(troca)] if troca else chave) + sufixo
            if candidato in mapa:
                return mapa[candidato]
        nome = "".join(p[:1].upper() + p[1:] for p in map(_sem_marcas, palavras))
        acentuado = "".join(p[:1].upper() + p[1:] for p in palavras)
        if acentuado != nome and re.fullmatch(r"[\w]+", acentuado):
            EXIBICAO.setdefault(nome, acentuado)
        return nome

    def nome_propriedade(self, palavras):
        palavras = self._sem_artigos(palavras)
        if not palavras:
            return None
        chave = _chave(" ".join(palavras))
        props = self.propriedades()
        if chave in props:
            return props[chave]
        if chave.endswith("am") and chave[:-1] in props:  # voam -> voa
            return props[chave[:-1]]
        return "_".join(_sem_marcas(p).lower() for p in palavras)

    @staticmethod
    def _dividir(palavras):
        grupos, atual = [], []
        for p in palavras:
            if p == "," or _sem_acento(p) == "e":
                if atual:
                    grupos.append(atual)
                atual = []
            else:
                atual.append(p)
        if atual:
            grupos.append(atual)
        return grupos

    @staticmethod
    def _prefixo(norm, opcoes):
        for opcao in opcoes:
            if tuple(norm[:len(opcao)]) == opcao:
                return len(opcao)
        return 0

    @staticmethod
    def _achar_relacao(norm, lex, fraco):
        melhor = None
        for i in range(len(norm)):
            for L in range(min(5, len(norm) - i), 0, -1):
                chave = tuple(norm[i:i + L])
                rel = lex.get(chave)
                if rel is None or "," in chave:
                    continue
                if (chave in FRACOS) != fraco:
                    continue
                if melhor is None or L > melhor[1] or (L == melhor[1] and i < melhor[0]):
                    melhor = (i, L, rel)
        return melhor

    # --- afirmações ---
    def afirmativa(self, texto):
        """Devolve {"fatos": [(s, r, o, negado)], "conf": c}, {"nova_relacao": ...} ou None."""
        toks = self.tokens(texto)
        norm = [_sem_acento(t) for t in toks]
        conf = 1.0
        mudou = True
        while mudou and norm:
            mudou = False
            for p in ENCHIMENTO + HEDGES:
                if tuple(norm[:len(p)]) == p:
                    if p in HEDGES:
                        conf = 0.6
                    toks, norm = toks[len(p):], norm[len(p):]
                    mudou = True
                    break
        while toks and toks[0] == ",":
            toks, norm = toks[1:], norm[1:]
        negado = any(n in NEGACOES for n in norm)
        pares = [(t, n) for t, n in zip(toks, norm) if n not in NEGACOES]
        toks, norm = [t for t, _ in pares], [n for _, n in pares]
        if len(toks) < 2:
            return None
        lex = self.lexico()
        achado = self._achar_relacao(norm, lex, False)
        if achado is None:
            # "é" sozinho (com acento) vale como é_um; "e" sem acento é conjunção e não conta
            fraco = self._achar_relacao(norm, lex, True)
            if fraco is not None and toks[fraco[0]] in ("é", "É"):
                achado = fraco
        if achado:
            i, L, rel = achado
            esquerda, direita = toks[:i], toks[i + L:]
            if esquerda and direita:
                sujeitos = [self.nome_entidade(g) for g in self._dividir(esquerda)]
                propriedade = rel in ("tem_propriedade", "não_tem_propriedade")
                objetos = [self.nome_propriedade(g) if propriedade else self.nome_entidade(g)
                           for g in self._dividir(direita)]
                fatos = [(s, rel, o, negado) for s in sujeitos for o in objetos if s and o]
                if fatos:
                    return {"fatos": fatos, "conf": conf}
        # verbo sem objeto que é uma propriedade conhecida: "o pinguim não voa"
        props = self.propriedades()
        for k in (3, 2, 1):
            if len(toks) <= k:
                continue
            chave = _chave(" ".join(norm[-k:]))
            if chave not in props and chave.endswith("am") and chave[:-1] in props:
                chave = chave[:-1]
            if chave in props:
                sujeitos = [self.nome_entidade(g) for g in self._dividir(toks[:-k])]
                fatos = [(s, "tem_propriedade", props[chave], negado) for s in sujeitos if s]
                if fatos:
                    return {"fatos": fatos, "conf": conf}
        # verbo desconhecido entre duas entidades conhecidas: propõe uma relação nova
        mapa = self.entidades()
        mencoes = []
        i = 0
        while i < len(norm):
            for L in range(min(3, len(norm) - i), 0, -1):
                if _chave(" ".join(norm[i:i + L])) in mapa and norm[i] not in ARTIGOS:
                    mencoes.append((i, i + L))
                    i += L - 1
                    break
            i += 1
        if len(mencoes) >= 2:
            (a1, a2), (b1, b2) = mencoes[0], mencoes[1]
            meio = [n for n in norm[a2:b1] if n not in ARTIGOS]
            if meio:
                s = self.nome_entidade(toks[a1:a2])
                o = self.nome_entidade(toks[b1:b2])
                return {"nova_relacao": (s, "_".join(meio), o, negado), "conf": conf}
        return None

    # --- interpretação completa ---
    def interpretar(self, texto):
        bruto = texto.strip()
        juntas = " ".join(_sem_acento(t) for t in self.tokens(bruto))
        for recurso in getattr(self.dialogo, "recursos", []):
            achado = recurso.interpretar(self, bruto, juntas)
            if achado:
                return achado
        m = re.match(r"^(?:carreg\w*|abr\w*|ler|leia|import\w*|usar|use)\s+"
                     r"(?:o\s+|a\s+)?(?:arquivo\s+|base\s+)?(.+)$", bruto, re.I)
        if m:
            caminho = m.group(1).strip().strip("\"'")
            if "." in caminho or os.path.exists(caminho):
                return {"tipo": "carregar", "caminho": caminho}
        m = re.match(r"^(?:export\w*|baix\w*)(?:\s+(?:a\s+)?(?:base\s+)?"
                     r"(?:em\s+|como\s+|para\s+|no\s+formato\s+|formato\s+)?(.+))?$", bruto, re.I)
        if m:
            arg = (m.group(1) or "").strip().strip("\"'")
            if "." in arg:
                return {"tipo": "exportar", "formato": None, "caminho": arg}
            formato = FORMATOS_EXPORTACAO.get(_sem_acento(arg), "txt" if not arg else None)
            if formato:
                return {"tipo": "exportar", "formato": formato, "caminho": None}
        m = re.match(r"^(?:salv\w*|grav\w*)(?:\s+(?:em\s+|como\s+|no\s+|para\s+)?(.+))?$",
                     bruto, re.I)
        if m and (not m.group(1) or "." in m.group(1)):
            return {"tipo": "salvar", "caminho": (m.group(1) or "").strip().strip("\"'") or None}
        toks = self.tokens(bruto)
        norm = [_sem_acento(t) for t in toks]
        juntas = " ".join(norm)
        if juntas in ("chega", "pare", "para", "chega de perguntas", "sem perguntas",
                      "para de perguntar", "pare de perguntar", "chega de perguntar",
                      "nao pergunte mais"):
            return {"tipo": "pausar"}
        if juntas in ("pode perguntar", "pergunte", "continue", "pode continuar",
                      "continua", "pergunta"):
            return {"tipo": "retomar"}
        if juntas in ("o que mudou", "o que mudou na base", "o que voce aprendeu",
                      "o que aprendeu", "mudancas", "o que foi alterado", "alteracoes"):
            return {"tipo": "mudou"}
        if juntas in ("resumo", "quanto voce sabe", "o que voce sabe", "status"):
            return {"tipo": "resumo"}
        if juntas in ("desfazer", "desfaca", "desfaz", "desfaz isso", "volte", "voltar", "volta"):
            return {"tipo": "desfazer"}
        if juntas in ("auditar", "audite", "audite a base", "auditoria", "revise a base",
                      "revisar a base", "verifique a base", "verificar a base", "confira a base"):
            return {"tipo": "auditar"}
        if juntas in ("aprenda padroes", "descubra padroes", "procure padroes", "encontre padroes",
                      "aprender padroes", "descubra regras", "aprenda regras", "aprender regras"):
            return {"tipo": "padroes"}
        if juntas in ("sugira", "sugestoes", "sugira fatos", "sugira algo", "o que mais",
                      "o que mais voce acha", "o que voce acha que falta", "o que falta"):
            return {"tipo": "sugerir"}
        m = (re.match(r"^como (?:o |a |os |as )?(.+?) (?:se relaciona|se relacionam|se liga|se ligam|"
                      r"se conecta|se conectam) (?:com |ao |a )?(?:o |a |os |as )?(.+)$", juntas)
             or re.match(r"^qual (?:e )?a relacao entre (?:o |a |os |as )?(.+?) e (?:o |a |os |as )?(.+)$",
                         juntas))
        if m:
            return {"tipo": "caminho", "a": self.nome_entidade(m.group(1).split()),
                    "b": self.nome_entidade(m.group(2).split())}
        m = re.match(r"^(?:mostre|mostra|desenhe|desenha|exiba|plote)(?: me)? (?:o |um )?"
                     r"(?:grafo|mapa|diagrama)(?: (?:de |do |da |sobre )(?:o |a )?(.+))?$", juntas)
        if m:
            return {"tipo": "grafo",
                    "entidade": self.nome_entidade(m.group(1).split()) if m.group(1) else None}
        n = self._prefixo(norm, [("me", "fale", "sobre"), ("me", "fale", "do"),
                                 ("me", "fale", "da"), ("fale", "sobre"), ("fale", "do"),
                                 ("fale", "da"), ("fale", "de"), ("me", "conte", "sobre"),
                                 ("conte", "sobre"), ("diga", "sobre"),
                                 ("o", "que", "voce", "sabe", "sobre"),
                                 ("o", "que", "voce", "sabe", "do"),
                                 ("o", "que", "voce", "sabe", "da"),
                                 ("quem", "e"), ("quem", "sao"), ("o", "que", "e"),
                                 ("o", "que", "sao")])
        if n and toks[n:]:
            return {"tipo": "descrever", "entidade": self.nome_entidade(toks[n:])}
        pergunta = bruto.endswith("?")
        n = self._prefixo(norm, [("por", "que"), ("porque",)])
        if n:
            analise = self.afirmativa(" ".join(toks[n:]))
            if analise and "fatos" in analise:
                s, r, o, neg = analise["fatos"][0]
                return {"tipo": "por_que", "fato": self._efetivo(s, r, o, neg)}
            return {"tipo": "nada"}
        lex = self.lexico()
        if norm[:1] == ["quem"] and len(norm) > 1:
            achado = self._achar_relacao(norm[1:], lex, False)
            if achado and achado[0] == 0 and toks[1 + achado[1]:]:
                objetos = self._dividir(toks[1 + achado[1]:])
                if objetos:
                    return {"tipo": "quem", "relacao": achado[2],
                            "objeto": self.nome_entidade(objetos[0])}
            return {"tipo": "nada"}
        if norm[:2] == ["o", "que"] and len(norm) > 2:
            resto, resto_n = toks[2:], norm[2:]
            achado = self._achar_relacao(resto_n, lex, False)
            if achado and achado[0] + achado[1] == len(resto_n) and achado[0] > 0:
                return {"tipo": "o_que", "sujeito": self.nome_entidade(resto[:achado[0]]),
                        "relacao": achado[2]}
            return {"tipo": "nada"}
        if norm[:1] == ["onde"] and len(norm) > 2 and norm[-1] in ("vive", "mora", "habita"):
            return {"tipo": "o_que", "sujeito": self.nome_entidade(toks[1:-1]),
                    "relacao": "habitat_principal"}
        analise = self.afirmativa(bruto)
        if analise and "nova_relacao" in analise:
            return {"tipo": "nova_relacao", "dados": analise["nova_relacao"],
                    "conf": analise["conf"]}
        if analise:
            if pergunta:
                s, r, o, neg = analise["fatos"][0]
                return {"tipo": "verificar", "fato": self._efetivo(s, r, o, neg)}
            return {"tipo": "afirmar", "fatos": analise["fatos"], "conf": analise["conf"]}
        return {"tipo": "nada"}

    def _efetivo(self, s, r, o, negado):
        """Aplica a negação: troca a relação pela oposta quando existe."""
        if negado:
            oposta = self.base.relacao_oposta(self.base.normalizar_relacao(r))
            if oposta:
                return (s, oposta, o, False)
        return (s, self.base.normalizar_relacao(r), o, negado)


# %% [20] Chat: leitura e gravação da base (txt, csv, json, pdf, pasta, zip)
# --- leitura e gravação da base de conhecimento ---
EXTENSOES = (".txt", ".md", ".csv", ".json", ".pdf")
_ALIAS = {"s": ("sujeito", "subject", "entidade", "s"),
          "r": ("relacao", "predicado", "verbo", "relation", "predicate", "r"),
          "o": ("objeto", "object", "valor", "o"),
          "c": ("confianca", "confidence", "peso", "c"),
          "f": ("origem", "fonte", "source"),
          "v": ("valido_ate", "validade", "ate", "vence_em", "vencimento")}
# linhas exportadas com estas origens são deduções: na releitura, não viram fato informado
ORIGENS_DEDUZIDAS = ("inferido", "deduzido", "classificador", "estimado")


def _numero(texto, padrao=1.0):
    try:
        return float(str(texto).replace(",", "."))
    except (TypeError, ValueError):
        return padrao


def _instalar(pacote):
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pacote], check=True, timeout=120)


def _importar_fitz():
    """PyMuPDF; se faltar (no Colab é comum), instala uma vez."""
    try:
        import fitz
    except ImportError:
        _instalar("pymupdf")
        import fitz
    return fitz


def ler_pdf(caminho):
    """Extrai o texto de um PDF com PyMuPDF (ou pypdf/PyPDF2, se for o que há instalado).
    Se nenhum estiver disponível, tenta instalar o PyMuPDF uma vez."""
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is None:
        for modulo in ("pypdf", "PyPDF2"):
            try:
                leitor = __import__(modulo).PdfReader(caminho)
            except ImportError:
                continue
            return "\n".join((pagina.extract_text() or "") for pagina in leitor.pages)
        fitz = _importar_fitz()
    with fitz.open(caminho) as documento:
        return "\n".join(pagina.get_text() for pagina in documento)


def _linhas_do_pdf(texto):
    """Junta as linhas quebradas de um parágrafo; linhas de tabela e declarações ficam à parte."""
    logicas, paragrafo = [], []

    def fechar():
        if paragrafo:
            logicas.append(" ".join(paragrafo))
            paragrafo.clear()
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            fechar()
        elif linha[:2] in ("+ ", "- ", "~ "):
            fechar()
            logicas.append(linha)
        elif len(linha.split()) <= 6 and not linha.endswith((".", "!", "?", ";", ":", ",")) \
                and linha.count("|") < 2 and not linha.startswith(("@", "#")):
            fechar()  # título ou frase curta sem ponto: fica numa linha própria
            logicas.append(linha)
        elif linha.count("|") >= 2 or linha.startswith(("@", "#")):
            fechar()
            logicas.append(linha)
        else:
            paragrafo.append(linha)
    fechar()
    return logicas


def _tripla(ia, s, r, o, conf=1.0, interprete=None, ate=None, fonte=None):
    s, r, o = str(s).strip(), str(r).strip(), str(o).strip()
    if not (s and r and o):
        return False
    if interprete is not None:
        # casa o nome com o já cadastrado (sem diferenciar maiúscula, acento ou plural)
        relacao = ia.base.normalizar_relacao(r)
        propriedade = relacao in ("tem_propriedade", "não_tem_propriedade")
        s = interprete.nome_entidade(s.split()) or s
        o = (interprete.nome_propriedade(o.split()) if propriedade
             else interprete.nome_entidade(o.split())) or o
    anterior = ia.base.proc_atual
    if fonte:
        # a linha traz a própria origem (arquivo gravado antes pelo Info): mantém
        arq, _, lin = fonte.partition(":")
        ia.base.proc_atual = dict(anterior, arquivo=arq, **({"linha": int(lin)} if lin.isdigit() else {}))
    try:
        ia.base.adicionar(s, r, o, conf, "observado", valido_ate=_data_iso(ate))
    finally:
        ia.base.proc_atual = anterior
    return True


def separar_validade(frase):
    """Separa 'o contrato vale até 31/12/2026' em (frase sem o prazo, 'AAAA-MM-DD')."""
    m = re.search(r"[,\s]+(?:at[eé]|v[aá]lid[oa]\s+at[eé]|vigente\s+at[eé])\s+"
                  r"(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\s*[.!;]?\s*$", frase, re.I)
    if not m:
        return frase, None
    data = _data_iso(m.group(1))
    return (frase[:m.start()], data) if data else (frase, None)


def _data_iso(texto):
    """Aceita AAAA-MM-DD ou DD/MM/AAAA e devolve AAAA-MM-DD (ou None)."""
    texto = str(texto or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", texto)
    if not m:
        m2 = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", texto)
        m = m2 and (None, m2.group(3), m2.group(2).zfill(2), m2.group(1).zfill(2))
        if not m:
            return None
        ano, mes, dia = m[1], m[2], m[3]
    else:
        ano, mes, dia = m.groups()
    try:
        return date(int(ano), int(mes), int(dia)).isoformat()
    except ValueError:
        return None


def _processar_linhas(ia, linhas, interprete, rotulo=""):
    """Cada linha é uma tripla separada por | ; tab ou vírgula, uma declaração @ ou frases."""
    lidas, ignoradas, diretivas = 0, [], 0
    for numero, linha in enumerate(linhas, 1):
        linha = linha.strip()
        if not linha or linha.startswith("#") or linha[:2] in ("+ ", "- ", "~ "):
            continue  # comentário ou linha de histórico de mudanças (relatório exportado)
        marca = f"{rotulo}:{numero}" if rotulo else str(numero)
        ia.base.proc_atual["linha"] = numero
        if linha.startswith("@"):
            if _diretiva(ia, linha):
                diretivas += 1
            else:
                ignoradas.append(marca)
            continue
        partes = None
        for sep in ("|", ";", "\t", ","):
            if linha.count(sep) >= 2:
                partes = [p.strip() for p in linha.split(sep)]
                break
        if partes and len(partes) >= 3:
            if _sem_acento(partes[0]) == "sujeito" and _sem_acento(partes[1]) == "relacao":
                continue
            conf = _numero(partes[3]) if len(partes) >= 4 and partes[3] else 1.0
            extras = dict(p.split("=", 1) for p in partes[4:] if "=" in p)
            lidas += _tripla(ia, partes[0], partes[1], partes[2], conf, interprete,
                             ate=extras.get("ate"), fonte=extras.get("fonte"))
            continue
        achou = False
        for frase in re.split(r"(?<=[.!;])\s+", linha):
            frase, ate = separar_validade(frase)
            analise = interprete.afirmativa(frase.strip(" .;!"))
            if analise and "fatos" in analise:
                achou = True
                for s, r, o, neg in analise["fatos"]:
                    s, r, o, neg = interprete._efetivo(s, r, o, neg)
                    if neg:
                        ia.base.rejeitados.add((s, r, o))
                    else:
                        ia.base.adicionar(s, r, o, analise["conf"], "observado", valido_ate=ate)
                    lidas += 1
        if not achou:
            ignoradas.append(marca)
    return {"lidas": lidas, "ignoradas": ignoradas, "diretivas": diretivas}


def _linhas_de_tabela_markdown(texto):
    """Das tabelas Markdown com cabeçalho `sujeito | relação | objeto`, devolve as linhas `s | r | o`.
    Outras tabelas e a prosa são ignoradas; linhas que começam com @ passam como declarações."""
    linhas, dentro = [], False
    for bruto in texto.splitlines():
        linha = bruto.strip()
        if linha.startswith("@"):
            linhas.append(linha)
            dentro = False
            continue
        if not linha.startswith("|"):
            dentro = False
            continue
        celulas = [c.strip().strip("`*") for c in linha.strip("|").split("|")]
        if re.fullmatch(r"[\s:|\-]+", linha):
            continue
        if (len(celulas) >= 3 and _sem_acento(celulas[0]) == "sujeito"
                and _sem_acento(celulas[1]) in ("relacao", "predicado", "verbo")):
            dentro = True
            continue
        if dentro and len(celulas) >= 3:
            partes = [x.strip() for x in celulas[2].split(",")]
            if len(partes) > 1 and all(0 < len(x.split()) <= 4 for x in partes):
                linhas += [" | ".join([celulas[0], celulas[1], x] + celulas[3:]) for x in partes]
            else:
                linhas.append(" | ".join(celulas))
    return linhas


def _importar_csv(ia, caminho, interprete):
    with open(caminho, encoding="utf-8-sig", errors="replace", newline="") as fp:
        texto = fp.read()
    uteis = [l for l in texto.splitlines() if l.strip() and not l.startswith("#")]
    if not uteis:
        return {"lidas": 0, "ignoradas": [], "diretivas": 0}
    sep = max(",;\t|", key=lambda d: uteis[0].count(d))
    linhas = list(csv.reader(io.StringIO("\n".join(uteis)), delimiter=sep))
    colunas = {"s": 0, "r": 1, "o": 2, "c": 3}
    nomes = [_sem_acento(c).strip() for c in linhas[0]]
    if any(n in _ALIAS["s"] for n in nomes) and any(n in _ALIAS["o"] for n in nomes):
        colunas = {k: next((i for i, n in enumerate(nomes) if n in _ALIAS[k]), None)
                   for k in _ALIAS}
        linhas = linhas[1:]
    if colunas == {"s": 0, "r": 1, "o": 2, "c": 3}:  # sem cabeçalho de triplas: pode ser tabela larga
        meta = detectar_tabela_larga([list(l) for l in csv.reader(io.StringIO("\n".join(uteis)),
                                                                  delimiter=sep)], ia)
        if meta:
            meta["arquivo"] = ia.base.proc_atual.get("arquivo") or os.path.basename(caminho)
            return {"lidas": importar_tabela(ia, meta), "ignoradas": [], "diretivas": 0,
                    "tabelas": [meta]}
    lidas, ignoradas = 0, []
    for numero, linha in enumerate(linhas, 1):
        def pos(k):
            c = colunas.get(k)
            return c if c is not None and c < len(linha) else None
        if pos("f") is not None and _sem_acento(linha[pos("f")]).strip() in ORIGENS_DEDUZIDAS:
            continue
        if pos("s") is None or pos("r") is None or pos("o") is None:
            sub = _processar_linhas(ia, [sep.join(linha)], interprete)
            lidas += sub["lidas"]
            ignoradas += [str(numero)] * len(sub["ignoradas"])
            continue
        conf = _numero(linha[pos("c")]) if pos("c") is not None and linha[pos("c")] else 1.0
        ia.base.proc_atual["linha"] = numero
        lidas += _tripla(ia, linha[pos("s")], linha[pos("r")], linha[pos("o")], conf, interprete,
                         ate=linha[pos("v")] if pos("v") is not None else None)
    return {"lidas": lidas, "ignoradas": ignoradas, "diretivas": 0}


def _mesclar_nativo(ia, dados):
    """Junta a base gravada por SistemaIA.salvar() à base atual (fatos firmes, regras, ontologia)."""
    b = ia.base
    ont = dados.get("ontologia", {})
    for r in ont.get("transitivas", []):
        b.declarar_transitiva(r)
    for r in ont.get("simetricas", []):
        b.declarar_simetrica(r)
    for r in ont.get("funcionais", []):
        b.declarar_funcional(r)
    for a, c in ont.get("inversas", {}).items():
        b.declarar_inversa(a, c)
    for a, c in ont.get("negacoes", {}).items():
        b.declarar_negacao(a, c)
    for a, c in ont.get("disjuntas", []):
        b.declarar_disjuntas(a, c)
    for d in dados.get("regras", []):
        regra = Regra(d["nome"], d["premissas"], d["conclusao"], origem=d["origem"],
                      alpha=d["alpha"], beta=d["beta"], metricas=d.get("metricas", {}))
        ia.motor.adicionar_regra(regra)
    lidas = 0
    for d in dados.get("fatos", []):
        if d.get("fonte") not in FONTES_FIRMES + ("classificador",):
            continue
        f = b.adicionar(d["s"], d["r"], d["o"], 0.5, d["fonte"], valido_ate=d.get("valido_ate"))
        if f is not None:
            f.alpha, f.beta = d["alpha"], d["beta"]
            if d.get("proc"):
                f.procedencias = d["proc"]
            lidas += 1
    for k in dados.get("rejeitados", []):
        b.rejeitados.add(tuple(k))
    for classe, obrig in dados.get("formas", {}).items():
        b.formas.setdefault(classe, [])
        b.formas[classe] += [r for r in obrig if r not in b.formas[classe]]
    b.vocabulario.update(dados.get("vocabulario", {}))
    b.config.update(dados.get("config", {}))
    b.rotulos += [tuple(x) for x in dados.get("rotulos", []) if tuple(x) not in b.rotulos]
    for fonte, (ac, er) in dados.get("fontes_stats", {}).items():
        par = b.fontes_stats.setdefault(fonte, [0, 0])
        par[0], par[1] = max(par[0], ac), max(par[1], er)
    return lidas


def _importar_json(ia, caminho, interprete):
    with open(caminho, encoding="utf-8-sig") as fp:
        dados = json.load(fp)
    lidas = 0
    if isinstance(dados, dict) and "fatos" in dados and "ontologia" in dados:
        return {"lidas": _mesclar_nativo(ia, dados), "ignoradas": [], "diretivas": 0}
    itens = None
    if isinstance(dados, dict):
        for chave in ("fatos", "triplas", "facts", "triples"):
            if isinstance(dados.get(chave), list):
                itens = dados[chave]
        if itens is None:
            # um objeto por entidade: {"Gato": {"é_um": "Mamifero", "caça": ["Rato"]}}
            for entidade, atributos in dados.items():
                if not isinstance(atributos, dict):
                    continue
                for relacao, valores in atributos.items():
                    for valor in (valores if isinstance(valores, list) else [valores]):
                        lidas += _tripla(ia, entidade, relacao, valor, 1.0, interprete)
            return {"lidas": lidas, "ignoradas": [], "diretivas": 0}
    else:
        itens = dados
    ignoradas = []
    for numero, item in enumerate(itens, 1):
        if isinstance(item, dict):
            chaves = {_sem_acento(k): v for k, v in item.items()}
            achado = {k: next((chaves[a] for a in _ALIAS[k] if a in chaves), None) for k in _ALIAS}
            if achado["f"] is not None and _sem_acento(str(achado["f"])).strip() in ORIGENS_DEDUZIDAS:
                continue
            if achado["s"] is not None and achado["r"] is not None and achado["o"] is not None:
                ia.base.proc_atual["linha"] = numero
                lidas += _tripla(ia, achado["s"], achado["r"], achado["o"],
                                 _numero(achado["c"]) if achado["c"] is not None else 1.0,
                                 interprete, ate=achado["v"])
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            conf = _numero(item[3]) if len(item) > 3 else 1.0
            lidas += _tripla(ia, item[0], item[1], item[2], conf, interprete)
            continue
        ignoradas.append(str(numero))
    return {"lidas": lidas, "ignoradas": ignoradas, "diretivas": 0}


def importar_arquivo(ia, caminho, interprete, rotulo=None):
    """Lê um arquivo (txt, csv, json ou pdf) para dentro de `ia`, anotando de onde cada fato veio."""
    anterior = ia.base.proc_atual
    ia.base.proc_atual = {"via": "arquivo", "arquivo": rotulo or os.path.basename(caminho)}
    try:
        return _importar_arquivo(ia, caminho, interprete)
    finally:
        ia.base.proc_atual = anterior


def _importar_arquivo(ia, caminho, interprete):
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".json":
        return _importar_json(ia, caminho, interprete)
    if ext == ".csv":
        return _importar_csv(ia, caminho, interprete)
    if ext == ".pdf":
        texto = ler_pdf(caminho)
        logicas = _linhas_do_pdf(texto)
        if FORMATO_RELATORIO in texto:
            # relatório gerado pelo Info: só a tabela e as declarações são dados, o resto é texto do relatório
            logicas = [l for l in logicas if l.startswith("@") or l.count("|") >= 2]
        info = _processar_linhas(ia, logicas, interprete)
        info["ignoradas"] = []  # em PDF, o que não vira fato (títulos, prosa) é ignorado em silêncio
        return info
    with open(caminho, encoding="utf-8-sig", errors="replace") as fp:
        if ext == ".md":
            texto = fp.read()
            for par in faq_do_markdown(texto):
                if par not in ia.base.faq:
                    ia.base.faq.append(par)
            info = _processar_linhas(ia, _linhas_de_tabela_markdown(texto), interprete)
            info["ignoradas"] = []  # a prosa do Markdown que não é tabela de fatos é ignorada em silêncio
            return info
        return _processar_linhas(ia, fp.read().splitlines(), interprete)


def importar_pasta(ia, pasta, interprete):
    """Lê todos os arquivos suportados de uma pasta e de suas subpastas."""
    total = {"lidas": 0, "ignoradas": [], "diretivas": 0, "arquivos": {}, "erros": [],
             "tabelas": []}
    caminhos = sorted(os.path.join(raiz, nome) for raiz, _, nomes in os.walk(pasta)
                      for nome in nomes if nome.lower().endswith(EXTENSOES))
    for caminho in caminhos:
        rotulo = os.path.relpath(caminho, pasta)
        try:
            info = importar_arquivo(ia, caminho, interprete, rotulo.replace("\\", "/"))
        except Exception as erro:
            total["erros"].append(f"{rotulo}: {erro}")
            continue
        total["lidas"] += info["lidas"]
        total["diretivas"] += info["diretivas"]
        total["tabelas"] += info.get("tabelas", [])
        total["ignoradas"] += [f"{rotulo}:{n}" for n in info["ignoradas"]]
        ext = os.path.splitext(caminho)[1].lower()
        total["arquivos"][ext] = total["arquivos"].get(ext, 0) + 1
    return total


def _diretiva(ia, linha):
    base = ia.base
    partes = linha.split()
    nome, args = partes[0].lower(), partes[1:]
    if nome == "@transitiva" and len(args) == 1:
        base.declarar_transitiva(args[0])
    elif nome == "@simetrica" and len(args) == 1:
        base.declarar_simetrica(args[0])
    elif nome == "@funcional" and len(args) == 1:
        base.declarar_funcional(args[0])
    elif nome == "@inversa" and len(args) == 2:
        base.declarar_inversa(args[0], args[1])
    elif nome == "@oposta" and len(args) == 2:
        base.declarar_negacao(args[0], args[1])
    elif nome == "@disjuntas" and len(args) == 2:
        base.declarar_disjuntas(args[0], args[1])
    elif nome == "@obrigatorio":
        m = re.match(r"^@obrigatorio\s+([^:]+):\s*(.+)$", linha)
        if not m:
            return False
        lista = base.formas.setdefault(m.group(1).strip(), [])
        for r in (x.strip() for x in m.group(2).split(",")):
            r = base.normalizar_relacao(r)
            if r and r not in lista:
                lista.append(r)
    elif nome == "@sinonimo":
        m = re.match(r"^@sinonimo\s+(.+?)\s*=\s*(\S+)\s*$", linha)
        if not m:
            return False
        base.vocabulario[_sem_acento(m.group(1)).strip()] = base.normalizar_relacao(m.group(2))
    elif nome == "@faq":
        m = re.match(r"^@faq\s+(.*?)\s*\|\s*(.+?)\s*\|\s*(.+)$", linha)
        if not m:
            return False
        if (m.group(1), m.group(2), m.group(3)) not in base.faq:
            base.faq.append((m.group(1), m.group(2), m.group(3)))
    elif nome == "@exibicao":
        m = re.match(r"^@exibicao\s+(\S+)\s*=\s*(\S+)\s*$", linha)
        if not m or _sem_marcas(m.group(2)) != m.group(1):
            return False
        EXIBICAO[m.group(1)] = m.group(2)
    elif nome == "@config" and len(args) >= 2:
        try:
            base.config[args[0]] = json.loads(" ".join(args[1:]))
        except ValueError:
            return False
    elif nome == "@fonte" and len(args) == 3 and all(a.isdigit() for a in args[1:]):
        base.fontes_stats[args[0]] = [int(args[1]), int(args[2])]
    elif nome == "@rotulo":
        m = re.match(r"^@rotulo\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([01])\s*$", linha)
        if not m:
            return False
        item = (m.group(1), base.normalizar_relacao(m.group(2)), m.group(3), int(m.group(4)))
        if item not in base.rotulos:
            base.rotulos.append(item)
    elif nome == "@regra":
        m = re.match(r"^@regra\s+([^:]+):\s*(.+?)\s*=>\s*(\([^)]*\))\s*([0-9.,]+)?\s*$", linha)
        if not m:
            return False
        premissas = [tuple(p.strip().strip("()").split()) for p in m.group(2).split("&")]
        conclusao = tuple(m.group(3).strip("()").split())
        if any(len(p) != 3 for p in premissas) or len(conclusao) != 3:
            return False
        conf = float(m.group(4).replace(",", ".")) if m.group(4) else 0.9
        ia.motor.adicionar_regra(Regra(m.group(1).strip(), premissas, conclusao, conf, "manual"))
    else:
        return False
    return True


def _declaracoes(ia):
    """Linhas @ com as propriedades das relações e as regras manuais ou induzidas."""
    b = ia.base
    linhas = [f"@oposta {pos} {neg}" for pos, neg in sorted(b.negacoes.items())]
    linhas += [f"@transitiva {r}" for r in sorted(b.transitivas)]
    linhas += [f"@simetrica {r}" for r in sorted(b.simetricas)]
    linhas += [f"@inversa {a} {c}" for a, c in sorted(b.inversas.items())]
    linhas += [f"@funcional {r}" for r in sorted(b.funcionais)]
    linhas += [f"@disjuntas {a} {c}" for a, c in b.disjuntas]
    linhas += [f"@obrigatorio {c}: {', '.join(rs)}" for c, rs in sorted(b.formas.items()) if rs]
    linhas += [f"@sinonimo {f} = {r}" for f, r in sorted(b.vocabulario.items())]
    usados = {x for f in b.fatos.values() for x in (f.s, f.o)}
    linhas += [f"@exibicao {n} = {a}" for n, a in sorted(EXIBICAO.items()) if n in usados]
    linhas += [f"@faq {i} | {p_} | {r_}" for i, p_, r_ in b.faq]
    linhas += [f"@config {k} {json.dumps(v, ensure_ascii=False)}" for k, v in sorted(b.config.items())]
    linhas += [f"@fonte {a} {ac} {er}" for a, (ac, er) in sorted(b.fontes_stats.items())]
    linhas += [f"@rotulo {s_} | {r_} | {o_} | {v}" for s_, r_, o_, v in b.rotulos]
    for regra in ia.motor.regras:
        if regra.origem in ("manual", "induzida"):
            prem = " & ".join("(" + " ".join(p) + ")" for p in regra.premissas)
            nome = regra.nome.replace(":", "_")
            linhas.append(f"@regra {nome}: {prem} => ({' '.join(regra.conclusao)}) "
                          f"{regra.confianca:.2f}")
    return linhas


def exportar_texto(ia, caminho):
    """Grava a base num arquivo de texto que importar_arquivo consegue ler de volta."""
    b = ia.base
    linhas = ["# Base de conhecimento gravada pelo prompt",
              "# sujeito | relação | objeto | confiança | fonte=arquivo:linha | ate=AAAA-MM-DD"
              ] + _declaracoes(ia)
    for f in sorted(b.fatos.values(), key=lambda x: x.chave):
        if f.fonte in FONTES_FIRMES:
            extra = ""
            p = next((p for p in f.procedencias if p.get("arquivo")), None)
            if p:
                extra += f" | fonte={p['arquivo']}" + (f":{p['linha']}" if p.get("linha") else "")
            if f.valido_ate:
                extra += f" | ate={f.valido_ate}"
            linhas.append(f"{f.s} | {f.r} | {f.o} | {f.confianca:.2f}{extra}")
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write("\n".join(linhas) + "\n")


FORMATO_RELATORIO = "formato info-relatorio-v1"

FORMATOS_EXPORTACAO = {
    "txt": "txt", "texto": "txt", "csv": "csv", "planilha": "csv", "excel": "csv",
    "json": "json", "pdf": "pdf", "relatorio": "pdf", "graphml": "graphml",
    "grafo": "graphml", "gephi": "graphml", "tudo": "tudo", "todos": "tudo",
    "completo": "tudo", "zip": "tudo", "pacote": "tudo"}


def _rgb(hexa):
    h = hexa.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def exportar_csv(ia, caminho):
    """Planilha com todos os fatos aceitos e a origem de cada um (informado ou deduzido)."""
    with open(caminho, "w", encoding="utf-8-sig", newline="") as fp:
        escritor = csv.writer(fp)
        escritor.writerow(["sujeito", "relacao", "objeto", "confianca", "origem", "valido_ate"])
        for f in sorted(ia.base.fatos.values(), key=lambda x: x.chave):
            if f.fonte in FONTES_FIRMES or f.confianca >= LIMIAR_ACEITE:
                escritor.writerow([f.s, f.r, f.o, f"{f.confianca:.2f}", f.fonte, f.valido_ate or ""])


def exportar_graphml(ia, caminho):
    """Grafo para Gephi, Cytoscape ou Neo4j: uma aresta por fato aceito."""
    g = nx.MultiDiGraph()
    for f in ia.base.fatos.values():
        if f.confianca >= LIMIAR_ACEITE:
            g.add_edge(f.s, f.o, relacao=f.r, confianca=round(f.confianca, 3), origem=f.fonte)
    nx.write_graphml(g, caminho)


def exportar_pdf(ia, caminho, mudancas=None):
    """Relatório com a identidade da marca. A tabela de fatos usa o mesmo formato que a
    leitura de PDF entende, então o relatório também serve como arquivo de entrada."""
    fitz = _importar_fitz()
    doc = fitz.open()
    acento = _rgb(MARCA["acento_claro"])
    est = {"pg": None, "y": 0}

    def nova():
        pg = doc.new_page()
        pg.insert_text((56, 42), MARCA["empresa"], fontname="hebo", fontsize=11, color=acento)
        pg.insert_text((56 + 62, 42), "| " + MARCA["ia"], fontname="helv", fontsize=11,
                       color=(0.45, 0.45, 0.45))
        pg.draw_line((56, 52), (539, 52), color=(0.88, 0.88, 0.88), width=0.6)
        est["pg"], est["y"] = pg, 82

    def escrever(texto, fonte="helv", tam=10.5, cor=(0.12, 0.12, 0.12), antes=0):
        if est["pg"] is None or est["y"] + antes > 780:
            nova()
        est["y"] += antes
        est["pg"].insert_text((56, est["y"]), texto, fontname=fonte, fontsize=tam, color=cor)
        est["y"] += tam * 1.5

    def paragrafo(texto, **kw):
        for parte in textwrap.wrap(texto, 92):
            escrever(parte, **kw)

    def tabela(texto):
        escrever(texto, fonte="cour", tam=min(9.0, 480 / (0.6 * max(len(texto), 1))))

    b = ia.base
    firmes = sorted((f for f in b.fatos.values() if f.fonte in FONTES_FIRMES), key=lambda x: x.chave)
    entidades = {f.s for f in firmes}
    classes = sorted({f.o for f in firmes if f.r == "é_um"})
    escrever("Base de conhecimento", fonte="hebo", tam=18, cor=acento)
    escrever(f"Exportada em {datetime.now():%d/%m/%Y %H:%M} por {MARCA['ia']}, da {MARCA['empresa']}",
             cor=(0.4, 0.4, 0.4))
    escrever(FORMATO_RELATORIO, fonte="cour", tam=8, cor=(0.6, 0.6, 0.6))
    escrever("Resumo", fonte="hebo", tam=13, antes=14)
    escrever(f"Fatos informados: {len(firmes)}")
    escrever(f"Fatos deduzidos: {len(b.fatos) - len(firmes)}")
    escrever(f"Entidades com fatos próprios: {len(entidades)}")
    if classes:
        paragrafo("Classes: " + ", ".join(classes))
    if mudancas and (mudancas["novos"] or mudancas["removidos"] or mudancas["alterados"]):
        escrever("O que mudou nesta sessão", fonte="hebo", tam=13, antes=14)
        for k in mudancas["novos"][:60]:
            tabela("+ " + frase_fato(*k))
        for k in mudancas["removidos"][:60]:
            tabela("- " + frase_fato(*k))
        for k, antes, depois in mudancas["alterados"][:60]:
            tabela(f"~ {frase_fato(*k)} ({antes:.0%} para {depois:.0%})")
    escrever("Regras e relações", fonte="hebo", tam=13, antes=14)
    for linha in _declaracoes(ia):
        tabela(linha)
    escrever("Fatos informados", fonte="hebo", tam=13, antes=14)
    paragrafo("Cada linha abaixo traz sujeito, relacao, objeto e confianca, separados por barra vertical.",
              cor=(0.4, 0.4, 0.4))
    tabela("sujeito | relacao | objeto | confianca")
    for f in firmes:
        tabela(f"{f.s} | {f.r} | {f.o} | {f.confianca:.2f}")
    doc.save(caminho)
    doc.close()


# %% [21] Chat: núcleo do diálogo (iniciativa, confronto, verificação)
# Cada recurso novo (fontes, tabelas, fuzzy...) é um plugin registrado nas células seguintes.

RECURSOS = []


def recurso(cls):
    """Registra um recurso plugável. O Dialogo instancia todos os registrados ao ser criado."""
    RECURSOS.append(cls)
    return cls


class Recurso:
    """Base dos recursos plugáveis. Cada método é opcional; o padrão não faz nada.

    interpretar : reconhece uma frase e devolve uma intenção (dict com "tipo"), ou None
    tratar      : executa uma intenção cujo tipo está em `tipos`; devolve linhas ou um subdiálogo
    perguntas   : candidatas a pergunta por iniciativa, com score, chave, gerador e motivo
    auditoria   : linhas extras para o comando auditar
    ao_abrir / ao_carregar / ao_salvar : ganchos do ciclo de vida
    apos_nova_relacao : subdiálogo opcional depois de criar uma relação nova
    sem_entendimento  : última chance de interpretar uma frase que ninguém entendeu
    reescrever : corrige a frase antes da interpretação (erro de digitação, pronomes)
    """
    nome = ""
    tipos = ()

    def __init__(self, dialogo):
        self.d = dialogo

    @property
    def ia(self):
        return self.d.ia

    @property
    def base(self):
        return self.d.ia.base

    def interpretar(self, itp, bruto, juntas):
        return None

    def tratar(self, intencao):
        return None

    def perguntas(self, foco):
        return []

    def auditoria(self):
        return []

    def ao_abrir(self):
        return []

    def ao_carregar(self, info):
        return []

    def ao_salvar(self, destino):
        return None

    def apos_nova_relacao(self, s, nome, o):
        return
        yield  # torna o método um gerador vazio

    def sem_entendimento(self, texto):
        return None

    def reescrever(self, texto):
        """Pode corrigir a frase antes de interpretá-la; devolve (texto, avisos)."""
        return texto, []


class Dialogo:
    """Estado da conversa. responder(texto) devolve as linhas que a IA escreve."""

    def __init__(self, ia=None):
        if ia is None:
            EXIBICAO.clear()      # conversa nova, sem os acentos de uma sessão anterior
        self.ia = ia or SistemaIA()
        self.diario = []          # eventos da sessão: fala, anotado, decisão, avaliação...
        self.ultima = []          # últimas linhas escritas pela IA
        self.ultimo_motivo = None  # por que a última pergunta por iniciativa foi feita
        self.hoje = None          # data de referência (None = hoje); útil em testes
        self.grau = 1.0           # firmeza da última resposta sim/não (ver grau_resposta)
        self.ult_fato = None      # último fato de que se falou (alimenta "mostre seu raciocínio")
        self._dicas_desfazer = 0  # quantas vezes já lembrou do "desfazer" após uma resposta livre
        self._meta = {}
        self.recursos = []
        self.rec = {}
        self.interprete = Interprete(self)
        self.ativo = None
        self.espera = None
        self.curioso = True
        self.perguntadas = set()
        self.fila = []
        self.recentes = []
        self.arquivo = None
        self.destino = None
        self.aprendidos = 0
        self.mudou = False
        self._avisou_fim = False
        self._pilha = []
        self._nao_empilhar = False
        self._base_inicial = {}
        self._emb, self._emb_fp = None, None
        self.recursos = [cls(self) for cls in RECURSOS]
        self.rec = {r.nome: r for r in self.recursos}

    def registrar(self, tipo, texto, **extra):
        """Anota um evento no diário da sessão (alimenta métricas e a ata)."""
        self.diario.append(dict(quando=datetime.now().isoformat(timespec="seconds"),
                                tipo=tipo, texto=texto, **extra))
        del self.diario[:-3000]

    def _hoje(self):
        return self.hoje or date.today()

    def _rotular(self, chave, verdade):
        """Guarda o julgamento da pessoa sobre uma dedução (insumo da calibração)."""
        b = self.ia.base
        b.rotulos = [x for x in b.rotulos if tuple(x[:3]) != tuple(chave)]
        b.rotulos.append((*chave, int(verdade)))

    @staticmethod
    def _arquivo_de(fato):
        return next((p["arquivo"] for p in fato.procedencias if p.get("arquivo")), None)

    # --- entrada e saída ---
    def abrir(self, arquivo=None):
        linhas = []
        if arquivo:
            linhas += self.carregar(arquivo)
        elif not self.ia.base.fatos:
            linhas += ["Ainda não tenho nenhuma base de conhecimento.",
                       "Anexe um arquivo (txt, md, csv, json, pdf ou zip), escreva carregar seguido do "
                       "nome do arquivo ou da pasta, ou me conte algo, por exemplo: o gato caça o rato.",
                       "Depois pergunte, por exemplo: o que você sabe sobre o gato?"]
            if os.path.exists("base_animais.txt"):
                linhas.append("Há um arquivo de exemplo neste ambiente: base_animais.txt")
        else:
            linhas.append(self._resumo_curto())
        for r in self.recursos:
            linhas += r.ao_abrir()
        linhas += self._iniciativa()
        return linhas

    def responder(self, texto):
        texto = texto.strip()
        if not texto:
            return []
        self.registrar("fala", texto[:300])
        antes = self._snapshot()
        self._nao_empilhar = False
        aceitas = self._aceitas()
        saida = self._responder(texto)
        self._empilhar(antes)
        mudadas = len(aceitas ^ self._aceitas())
        if aceitas and mudadas > 1 and not self._nao_empilhar:
            saida.append(f"Isso mudou {mudadas} conclusões minhas. Escreva o que mudou para ver.")
        self.ultima = saida
        return saida

    def _aceitas(self):
        return {k for k, f in self.ia.base.fatos.items() if f.confianca >= LIMIAR_ACEITE}

    def _explicar_motivo(self):
        if not self.ultimo_motivo:
            return ["Ainda não fiz nenhuma pergunta por iniciativa própria nesta conversa."]
        return [f"Perguntei porque {self.ultimo_motivo}."]

    def _conf_txt(self, c):
        """Confiança escrita para a pessoa: percentual e, no modo fuzzy, também um termo."""
        texto = f"{c:.0%}"
        if self.ia.base.config.get("rotulos_fuzzy"):
            texto += f", {termo_fuzzy(c)}"
        return texto

    def _responder(self, texto):
        saida = []
        respondia_livre = self.ativo is not None and self.espera == "livre"
        if self.ativo is not None and _sem_acento(texto).strip(" .!") in (
                "cancelar", "cancela", "deixa pra la", "deixa", "esquece", "pular", "pula", "outra coisa"):
            self.ativo, self.espera = None, None
            return ["Tudo bem, deixei essa pergunta de lado. Pode continuar do ponto que preferir."]
        if self.ativo is not None and self.interprete.interpretar(texto)["tipo"] == "motivo":
            return self._explicar_motivo() + _lista(getattr(self, "_pergunta_atual", ""))
        if self.ativo is not None:
            if self.espera == "sim_nao":
                tipo = classificar_resposta(texto)
                if tipo == "outro":
                    self.ativo = None
                    saida += self._livre(texto)
                else:
                    self.grau = grau_resposta(texto)
                    try:
                        saida += self._avancar(self.ativo, tipo)
                    finally:
                        self.grau = 1.0
            elif self._e_pergunta_ou_comando(texto):
                self.ativo = None  # a pessoa perguntou ou pediu algo em vez de responder
                saida += self._livre(texto)
            else:
                saida += self._avancar(self.ativo, texto)
        else:
            saida += self._livre(texto)
        if self.ativo is None:
            saida += self._iniciativa()
        saida += self._autosalvar()
        if respondia_livre and self._dicas_desfazer < 2 and any(l.startswith("Anotei") for l in saida):
            self._dicas_desfazer += 1
            saida.append("(Se não era isso que eu devia anotar, escreva desfazer.)")
        return saida

    def _e_pergunta_ou_comando(self, texto):
        if texto.rstrip().endswith("?"):
            return True
        tipo = self.interprete.interpretar(texto)["tipo"]
        extras = tuple(t for r in self.recursos for t in r.tipos)
        return tipo in ("carregar", "salvar", "pausar", "retomar", "resumo", "descrever",
                        "quem", "o_que", "por_que", "verificar", "exportar", "mudou", "desfazer",
                        "auditar", "padroes", "sugerir", "caminho", "grafo") + extras

    def anexar(self, caminho):
        antes = self._snapshot()
        saida = self.carregar(caminho)
        if self.ativo is None:
            saida += self._iniciativa()
        saida += self._autosalvar()
        self._empilhar(antes)
        return saida

    def _avancar(self, gerador, valor):
        antes = self.ativo
        try:
            item = next(gerador) if valor is None else gerador.send(valor)
        except StopIteration as fim:
            if self.ativo is antes:   # um esclarecimento pode ter instalado outra pergunta
                self.ativo, self.espera = None, None
            return _lista(fim.value)
        texto, espera = item
        self.ativo, self.espera = gerador, espera
        self._pergunta_atual = texto.split("\n")[-1]
        return _lista(texto)

    def _iniciar(self, gerador):
        return self._avancar(gerador, None)

    def _delegar(self, sub, acum):
        """Repassa as perguntas de um subdiálogo, junto das linhas já acumuladas."""
        try:
            item = next(sub)
        except StopIteration as fim:
            acum.extend(_lista(fim.value))
            return
        while True:
            texto, espera = item
            resposta = yield ("\n".join(acum + [texto]), espera)
            acum.clear()
            try:
                item = sub.send(resposta)
            except StopIteration as fim:
                acum.extend(_lista(fim.value))
                return

    # --- tratamento do texto livre ---
    def _lembrar(self, *nomes):
        for nome in nomes:
            if nome in self.recentes:
                self.recentes.remove(nome)
            self.recentes.insert(0, nome)
        del self.recentes[8:]

    def _livre(self, texto):
        avisos = []
        for r in self.recursos:
            texto, extra = r.reescrever(texto)
            avisos += extra
        return avisos + self._livre_interno(texto)

    def _livre_interno(self, texto):
        intencao = self.interprete.interpretar(texto)
        tipo = intencao["tipo"]
        for r in self.recursos:
            if tipo in r.tipos:
                resultado = r.tratar(intencao)
                if resultado is not None:
                    return self._iniciar(resultado) if inspect.isgenerator(resultado) else resultado
        if tipo == "carregar":
            return self.carregar(intencao["caminho"])
        if tipo == "salvar":
            return self.salvar(intencao["caminho"])
        if tipo == "pausar":
            self.curioso = False
            return ["Certo, paro de perguntar. Escreva pode perguntar quando quiser que eu volte."]
        if tipo == "retomar":
            self.curioso, self._avisou_fim = True, False
            return ["Certo."]
        if tipo == "resumo":
            return self._resumo()
        if tipo == "exportar":
            return self._exportar(intencao["formato"], intencao["caminho"])
        if tipo == "mudou":
            return self._o_que_mudou()
        if tipo == "desfazer":
            return self._desfazer()
        if tipo == "auditar":
            return self._auditar()
        if tipo == "padroes":
            return self._padroes()
        if tipo == "sugerir":
            return self._sugerir()
        if tipo == "caminho":
            return self._caminho(intencao["a"], intencao["b"])
        if tipo == "grafo":
            return self._grafo(intencao["entidade"])
        if tipo == "descrever":
            return self._descrever(intencao["entidade"])
        if tipo == "quem":
            return self._quem(intencao["relacao"], intencao["objeto"])
        if tipo == "o_que":
            return self._o_que(intencao["sujeito"], intencao["relacao"])
        if tipo == "por_que":
            return self._por_que(*intencao["fato"][:3])
        if tipo == "verificar":
            s, r, o, neg = intencao["fato"]
            self._lembrar(s)
            return self._iniciar(self._verificar(s, r, o, neg))
        if tipo == "afirmar":
            self._lembrar(*[f[0] for f in intencao["fatos"]])
            return self._iniciar(self._afirmar_varios(intencao["fatos"], intencao["conf"],
                                                      intencao.get("ate")))
        if tipo == "nova_relacao":
            return self._iniciar(self._nova_relacao(*intencao["dados"], intencao["conf"]))
        if classificar_resposta(texto) != "outro":
            return ["No momento não tenho nenhuma pergunta aberta. Pode me contar algo ou perguntar."]
        for r in self.recursos:
            resultado = r.sem_entendimento(texto)
            if resultado is not None:
                return self._iniciar(resultado) if inspect.isgenerator(resultado) else resultado
        return ["Não entendi essa frase. Escreva ajuda para ver o que sei fazer.",
                self._sugestao_de_frase()]

    def _sugestao_de_frase(self):
        """Exemplos de frases que o Info entende, montados com o assunto mais citado da base."""
        b = self.ia.base
        classes = {f.o for f in b.fatos.values() if f.r in ("é_um", "subclasse_de")}
        contagem = Counter(f.s for f in b.fatos.values()
                           if f.fonte in FONTES_FIRMES and f.s not in classes)
        if not contagem:
            return ("Tente algo como: o gato caça o rato, o pinguim não voa, "
                    "ou uma pergunta como: quem caça o salmão?")
        e = _legivel(contagem.most_common(1)[0][0])
        return (f"Tente reformular, por exemplo: o que você sabe sobre {e}? "
                f"ou: e se {e} voa? Também aceito fatos novos, como: o gato caça o rato.")

    # --- arquivos ---
    def carregar(self, caminho):
        caminho = caminho.strip().strip("\"'")
        baixado = None
        if re.match(r"^https?://", caminho, re.I):
            baixado = self._baixar(caminho)
            if isinstance(baixado, list):
                return baixado
            caminho = baixado
        if not os.path.exists(caminho):
            return [f"Não encontrei '{caminho}'. Confira o nome ou anexe o arquivo."]
        raiz = caminho.rstrip("/\\")
        nome, ext = os.path.splitext(raiz)
        try:
            if ext.lower() == ".zip":
                destino_zip = nome
                while os.path.exists(destino_zip):
                    destino_zip += "_1"
                with zipfile.ZipFile(caminho) as pacote:
                    if sum(i.file_size for i in pacote.infolist()) > 20 * LIMITE_DOWNLOAD:
                        return [f"O conteúdo de '{caminho}' é grande demais para abrir aqui."]
                    pacote.extractall(destino_zip)
                raiz, nome, ext = destino_zip, destino_zip, ""
            if os.path.isdir(raiz):
                info = importar_pasta(self.ia, raiz, self.interprete)
                nome, ext = raiz, ""
            else:
                info = importar_arquivo(self.ia, raiz, self.interprete)
        except Exception as erro:
            return [f"Não consegui ler '{caminho}': {erro}"]
        conflitos = self.ia.motor.verificar_consistencia(resolver=False)
        self.ia.pensar(verbose=False)
        self.arquivo = raiz
        self.destino = f"{nome}_atualizada{'.json' if ext.lower() == '.json' else '.txt'}"
        self.perguntadas.clear()
        self._avisou_fim = False
        self._base_inicial = self._conf_firmes()
        for tipo, a, b, _ in conflitos:
            self.fila.append(self._q_conflito(a, b))
        base = self.ia.base
        entidades = {f.s for f in base.fatos.values() if f.fonte in FONTES_FIRMES}
        classes = sorted({f.o for f in base.fatos.values() if f.r == "é_um"
                          and f.fonte in FONTES_FIRMES})
        if "arquivos" in info:
            tipos = ", ".join(f"{n} {e[1:]}" for e, n in sorted(info["arquivos"].items()))
            linhas = [f"Li a pasta '{os.path.basename(raiz)}' ({tipos or 'nenhum arquivo suportado'}): "
                      f"{info['lidas']} fatos sobre {len(entidades)} entidades."]
            linhas += [f"Não consegui ler {e}" for e in info["erros"]]
        else:
            linhas = [f"Li '{os.path.basename(raiz)}': {info['lidas']} fatos "
                      f"sobre {len(entidades)} entidades."]
        linhas.append(f"O que eu aprender fica gravado em '{self.destino}'; os arquivos "
                      "originais não são alterados.")
        if classes:
            resto = f" e mais {len(classes) - 8}" if len(classes) > 8 else ""
            linhas.append("Classes que conheço: " + ", ".join(classes[:8]) + resto + ".")
        if info["diretivas"]:
            linhas.append(f"Também li {info['diretivas']} declarações de regras e relações.")
        if info["lidas"] == 0 and "arquivos" not in info and os.path.getsize(raiz) > 0:
            linhas.append("Não reconheci nenhum fato neste arquivo. Aceito linhas como "
                          "'gato | é_um | mamífero', CSV com sujeito, relação e objeto, ou JSON "
                          "gravado pelo Info.")
        if info["ignoradas"]:
            linhas.append(f"Não entendi {len(info['ignoradas'])} linha(s): "
                          + ", ".join(str(n) for n in info["ignoradas"][:5])
                          + ("..." if len(info["ignoradas"]) > 5 else "."))
        if conflitos:
            linhas.append(f"Encontrei {len(conflitos)} ponto(s) em que a base se contradiz. "
                          "Vou perguntar sobre eles.")
        if baixado:
            linhas.insert(0, f"Baixei {os.path.basename(baixado)}.")
        self.registrar("carga", f"Li {os.path.basename(raiz)}: {info['lidas']} fatos.")
        for r in self.recursos:
            linhas += r.ao_carregar(info)
        return linhas

    def salvar(self, caminho=None):
        destino = caminho or self.destino or "base_conhecimento.txt"
        try:
            if destino.lower().endswith(".json"):
                with redirect_stdout(io.StringIO()):
                    self.ia.salvar(destino)
            else:
                exportar_texto(self.ia, destino)
        except OSError as erro:
            return [f"Não consegui gravar em '{destino}': {erro}"]
        for r in self.recursos:
            r.ao_salvar(destino)
        return [f"Gravei a base em '{destino}'."]

    def _autosalvar(self):
        if not self.mudou:
            return []
        self.mudou = False
        primeira = self.destino is None
        destino = self.destino or "base_conhecimento.txt"
        self.destino = destino
        self.salvar(destino)
        return [f"(Estou gravando o que aprendo em '{destino}'.)"] if primeira else []

    # --- consultas ---
    def estado(self):
        """(nome do arquivo carregado ou None, quantidade de fatos informados)."""
        firmes = sum(1 for f in self.ia.base.fatos.values() if f.fonte in FONTES_FIRMES)
        return (os.path.basename(self.arquivo) if self.arquivo else None), firmes

    def _resumo_curto(self):
        b = self.ia.base
        firmes = [f for f in b.fatos.values() if f.fonte in FONTES_FIRMES]
        return f"Tenho {len(firmes)} fatos informados e {len(b.fatos) - len(firmes)} deduzidos."

    def _resumo(self):
        linhas = [self._resumo_curto()]
        if self.aprendidos:
            linhas.append(f"Nesta conversa aprendi {self.aprendidos} fato(s) novo(s).")
        return linhas

    # --- histórico: snapshots, desfazer e "o que mudou" ---
    def _conf_firmes(self):
        return {f.chave: f.confianca for f in self.ia.base.fatos.values()
                if f.fonte in FONTES_FIRMES}

    def _snapshot(self):
        b = self.ia.base
        return {"fatos": {f.chave: (f.alpha, f.beta, f.fonte, f.valido_ate) for f in b.fatos.values()
                          if f.fonte in FONTES_FIRMES + ("classificador",)},
                "rejeitados": set(b.rejeitados),
                "extras": copy.deepcopy((b.vocabulario, b.formas, b.rotulos, b.config)),
                "regras": {r.nome: (r.alpha, r.beta) for r in self.ia.motor.regras
                           if r.origem != "estrutural"}}

    @staticmethod
    def _assinatura(snap):
        return (frozenset((k, round(a, 3), round(be, 3)) for k, (a, be, _, _v) in snap["fatos"].items()),
                frozenset(snap["rejeitados"]),
                repr(snap["extras"]),
                frozenset((n, round(a, 3), round(be, 3)) for n, (a, be) in snap["regras"].items()))

    def _empilhar(self, antes):
        if self._nao_empilhar:
            return
        if self._assinatura(antes) != self._assinatura(self._snapshot()):
            self._pilha.append(antes)
            del self._pilha[:-30]

    def _restaurar(self, snap):
        b = self.ia.base
        b.fatos.clear()
        b._idx_s.clear()
        b._idx_r.clear()
        b._idx_o.clear()
        b.rejeitados = set()
        for (s, r, o), (a, be, fonte, validade) in snap["fatos"].items():
            f = b.adicionar(s, r, o, 0.5, fonte, valido_ate=validade)
            f.alpha, f.beta = a, be
        b.rejeitados = set(snap["rejeitados"])
        b.vocabulario, b.formas, b.rotulos, b.config = copy.deepcopy(snap["extras"])
        regras = [r for r in self.ia.motor.regras
                  if r.origem == "estrutural" or r.nome in snap["regras"]]
        for r in regras:
            if r.nome in snap["regras"]:
                r.alpha, r.beta = snap["regras"][r.nome]
        self.ia.motor.regras = regras
        self.ia.motor._assinaturas = {r.assinatura(): r for r in regras}
        self.ia.pensar(verbose=False)

    def _desfazer(self):
        self._nao_empilhar = True
        if not self._pilha:
            return ["Não há nada para desfazer."]
        antes = set(self._conf_firmes())
        self._restaurar(self._pilha.pop())
        depois = set(self._conf_firmes())
        self.ativo, self.espera = None, None
        self.mudou = True
        return [f"Desfiz a última alteração: {len(antes - depois)} fato(s) saíram e "
                f"{len(depois - antes)} voltaram."]

    def _mudancas(self):
        atual, base = self._conf_firmes(), self._base_inicial
        return {"novos": sorted(k for k in atual if k not in base),
                "removidos": sorted(k for k in base if k not in atual),
                "alterados": sorted((k, base[k], atual[k]) for k in atual
                                    if k in base and abs(atual[k] - base[k]) >= 0.15)}

    def _o_que_mudou(self):
        m = self._mudancas()
        if not (m["novos"] or m["removidos"] or m["alterados"]):
            return ["Ainda não mudei nada na base desde que a carreguei."]
        linhas = []

        def listar(titulo, itens, formato):
            if itens:
                linhas.append(f"{titulo} ({len(itens)}):")
                linhas.extend("  " + formato(i) for i in itens[:12])
                if len(itens) > 12:
                    linhas.append(f"  e mais {len(itens) - 12}.")
        listar("Fatos novos", m["novos"], lambda k: frase_fato(*k))
        listar("Fatos removidos", m["removidos"], lambda k: frase_fato(*k))
        listar("Confiança alterada", m["alterados"],
               lambda t: f"{frase_fato(*t[0])} ({t[1]:.0%} para {t[2]:.0%})")
        linhas.append("Escreva exportar tudo para levar a base com essas mudanças.")
        return linhas

    # --- exportação ---
    def _stem(self):
        base = os.path.splitext(self.destino or "base_conhecimento.txt")[0]
        return re.sub(r"_atualizada$", "", base)

    def _escrever_formato(self, formato, caminho, mudancas):
        if formato == "txt":
            exportar_texto(self.ia, caminho)
        elif formato == "csv":
            exportar_csv(self.ia, caminho)
        elif formato == "json":
            with redirect_stdout(io.StringIO()):
                self.ia.salvar(caminho)
        elif formato == "pdf":
            exportar_pdf(self.ia, caminho, mudancas)
        elif formato == "graphml":
            exportar_graphml(self.ia, caminho)

    def _exportar(self, formato, caminho=None):
        if not self.ia.base.fatos:
            return ["Ainda não há nada para exportar."]
        if caminho:
            ext = os.path.splitext(caminho)[1].lower()
            formato = {".txt": "txt", ".csv": "csv", ".json": "json", ".pdf": "pdf",
                       ".graphml": "graphml", ".zip": "tudo"}.get(ext)
            if formato is None:
                return [f"Não sei exportar para '{ext}'. Use txt, csv, json, pdf, graphml ou zip."]
        stem, mud = self._stem(), self._mudancas()
        try:
            if formato == "tudo":
                destino_zip = caminho or stem + "_exportacao.zip"
                pasta = os.path.splitext(destino_zip)[0]
                os.makedirs(pasta, exist_ok=True)
                nome = os.path.basename(stem) + "_exportada"
                arquivos = []
                for ext in ("txt", "csv", "json", "pdf", "graphml"):
                    alvo = os.path.join(pasta, f"{nome}.{ext}")
                    self._escrever_formato(ext, alvo, mud)
                    arquivos.append(alvo)
                with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as pacote:
                    for alvo in arquivos:
                        pacote.write(alvo, os.path.basename(alvo))
                principal = destino_zip
                resumo = (f"Exportei {len(arquivos)} arquivos (txt, csv, json, pdf e graphml) "
                          f"em '{pasta}' e compactei em '{destino_zip}'.")
            else:
                principal = caminho or f"{stem}_exportada.{formato}"
                self._escrever_formato(formato, principal, mud)
                resumo = f"Exportei a base em '{principal}'."
        except Exception as erro:
            return [f"Não consegui exportar: {erro}"]
        return [resumo, f"[[baixar:{principal}]]"]

    # --- carga por URL ---
    @staticmethod
    def _normalizar_url(url):
        m = re.match(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)", url)
        if m:
            gid = re.search(r"[#&?]gid=(\d+)", url)
            return (f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
                    + (f"&gid={gid.group(1)}" if gid else ""))
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
        if m:
            return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
        return url

    def _baixar(self, url):
        original, url = url, self._normalizar_url(url)
        try:
            pedido = urllib.request.Request(url, headers={"User-Agent": MARCA["ia"] + "-" + MARCA["empresa"]})
            with urllib.request.urlopen(pedido, timeout=20) as resposta:
                dados = resposta.read(LIMITE_DOWNLOAD + 1)
                tipo = resposta.headers.get("Content-Type", "")
        except Exception as erro:
            return [f"Não consegui baixar '{original}': {erro}"]
        if len(dados) > LIMITE_DOWNLOAD:
            return [f"O arquivo tem mais de {LIMITE_DOWNLOAD // 1048576} MB, acima do que consigo ler aqui."]
        nome = os.path.basename(urllib.parse.urlparse(url).path) or "download"
        if "docs.google.com" in url:
            nome = "planilha_" + re.sub(r"\W", "", url.split("/d/")[1][:8]) + ".csv"
        ext = os.path.splitext(nome)[1].lower()
        if ext not in EXTENSOES + (".zip",):
            ext = next((e for k, e in (("csv", ".csv"), ("json", ".json"), ("pdf", ".pdf"),
                                       ("zip", ".zip")) if k in tipo), ".txt")
            nome = (os.path.splitext(nome)[0] or "download") + ext
        os.makedirs("downloads_info", exist_ok=True)
        caminho = os.path.join("downloads_info", nome)
        with open(caminho, "wb") as fp:
            fp.write(dados)
        return caminho

    # --- auditoria, padrões, sugestões, caminho e grafo ---
    def _q_mesclar(self, a, c):
        resp = yield (f"Parece que '{a}' e '{c}' são a mesma coisa. Junto as duas?", "sim_nao")
        if resp != "sim":
            return ["Certo, mantenho separadas."]
        b = self.ia.base
        pesos = {x: len(b.consultar(s=x)) + len(b.consultar(o=x)) for x in (a, c)}
        manter, sair = (a, c) if pesos[a] >= pesos[c] else (c, a)
        for f in list(b.fatos.values()):
            if f.fonte in FONTES_FIRMES and sair in (f.s, f.o):
                novo = (manter if f.s == sair else f.s, f.r, manter if f.o == sair else f.o)
                b.remover(f.chave)
                if novo[0] != novo[2]:
                    b.adicionar(*novo, f.confianca, f.fonte)
        self.ia.pensar(verbose=False)
        self.mudou = True
        return [f"Juntei '{sair}' em '{manter}'."]

    def _auditar(self):
        b = self.ia.base
        firmes = [f for f in b.fatos.values() if f.fonte in FONTES_FIRMES]
        if not firmes:
            return ["Não há nada para auditar ainda."]
        linhas, achados = ["Auditoria da base:"], 0
        classes = ({f.o for f in b.fatos.values() if f.r in ("é_um", "subclasse_de")}
                   | {f.s for f in b.fatos.values() if f.r == "subclasse_de"})
        com_classe = {f.s for f in b.fatos.values() if f.r == "é_um"}
        sem_classe = sorted({f.s for f in firmes} - classes - com_classe)
        if sem_classe:
            achados += 1
            linhas.append(f"  {len(sem_classe)} entidade(s) sem classe: " + ", ".join(sem_classe[:8])
                          + ("..." if len(sem_classe) > 8 else "."))
        conflitos = self.ia.motor.verificar_consistencia(resolver=False)
        for _, a, c, _ in conflitos:
            self.fila.append(self._q_conflito(a, c))
        if conflitos:
            achados += 1
            linhas.append(f"  {len(conflitos)} contradição(ões) a resolver.")
        nomes = sorted(e for e in b.entidades() if len(e) >= 4)[:300]
        pares = [(x, y) for i, x in enumerate(nomes) for y in nomes[i + 1:]
                 if x[0].lower() == y[0].lower()
                 and difflib.SequenceMatcher(None, _chave(x), _chave(y)).ratio() >= 0.88]
        for x, y in pares[:3]:
            self.fila.append(self._q_mesclar(x, y))
        if pares:
            achados += 1
            linhas.append("  Possíveis duplicatas: " + "; ".join(f"{x} e {y}" for x, y in pares[:4]) + ".")
        fracos = [f for f in firmes if f.confianca < 0.7]
        if fracos:
            achados += 1
            linhas.append(f"  {len(fracos)} fato(s) informado(s) com confiança abaixo de 70%.")
        uso = Counter(f.r for f in firmes)
        raras = sorted(r for r, n in uso.items()
                       if n == 1 and r not in FORMAS_RELACAO and r not in ("é_um", "subclasse_de"))
        if raras:
            achados += 1
            linhas.append("  Relações usadas uma única vez (pode ser erro de digitação): "
                          + ", ".join(raras[:6]) + ".")
        com_props = {f.s for f in b.fatos.values() if f.r == "tem_propriedade"}
        vazias = sorted({f.o for f in firmes if f.r == "é_um"} - com_props)
        if vazias:
            achados += 1
            linhas.append("  Classes com exemplos, mas sem nenhuma característica própria: "
                          + ", ".join(vazias[:6]) + ".")
        fracas = [r.nome for r in self.ia.motor.regras
                  if r.origem in ("manual", "induzida") and r.confianca < 0.5]
        if fracas:
            achados += 1
            linhas.append("  Regras com confiança abaixo de 50%: " + ", ".join(fracas[:5]) + ".")
        for r in self.recursos:
            extra = r.auditoria()
            if extra:
                achados += 1
                linhas.extend(extra)
        if not achados:
            return ["Auditoria da base: não encontrei problemas."]
        if self.fila:
            linhas.append(f"Vou perguntar sobre {len(self.fila)} ponto(s).")
        return linhas

    def _padrao_em_frase(self, regra):
        letras = {}

        def nome(t):
            if e_var(t):
                letras.setdefault(t, "XYZWUV"[len(letras) % 6])
                return letras[t]
            return t

        def frase(p):
            return frase_fato(nome(p[0]), p[1], nome(p[2]))
        return ("Quando " + " e ".join(frase(p) for p in regra.premissas)
                + ", então " + frase(regra.conclusao) + ".")

    def _q_regra(self, regra):
        m = regra.metricas
        resp = yield (f"{self._padrao_em_frase(regra)} Vi isso em {m.get('suporte', '?')} de "
                      f"{m.get('corpo', '?')} casos. Esse padrão procede?", "sim_nao")
        if resp == "sim":
            regra.alpha += 3
            self.mudou = True
            return ["Certo, confirmei o padrão."]
        if resp == "nao":
            regra.beta += 3
            self.mudou = True
            self.ia.pensar(verbose=False)
            return ["Entendi. Reduzi a confiança nesse padrão."]
        return ["Tudo bem, mantenho o padrão com a confiança que calculei."]

    def _padroes(self):
        firmes = [f for f in self.ia.base.fatos.values() if f.fonte in FONTES_FIRMES]
        if len(firmes) > LIMITE_INDUCAO:
            return [f"A base tem {len(firmes)} fatos informados; acima de {LIMITE_INDUCAO} a busca "
                    "de padrões fica lenta demais para o Colab. Carregue uma parte da base."]
        if len(firmes) < 8:
            return ["Ainda tenho poucos fatos para achar padrões."]
        novas = self.ia.indutor.induzir()
        self.ia.pensar(verbose=False)
        if not novas:
            return ["Não encontrei padrões novos. Cada padrão precisa de pelo menos 3 exemplos."]
        self.mudou = True
        for regra in novas[:5]:
            self.fila.append(self._q_regra(regra))
        return [f"Encontrei {len(novas)} padrão(ões) nos dados e eles já estão valendo. "
                f"Vou pedir sua opinião sobre {min(5, len(novas))}."]

    def _q_sugestao(self, s, r, o, p):
        resp = yield (f"Pelo padrão da base, acho que talvez {frase_fato(s, r, o)} "
                      f"(é só um palpite do modelo, pontuação {p:.2f} de 1). Procede?", "sim_nao")
        acum = []
        if resp in ("sim", "nao"):
            self.registrar("sugestao", frase_fato(s, r, o), aceita=resp == "sim")
        if resp == "sim":
            yield from self._delegar(self._afirmar(s, r, o, False, 1.0), acum)
            return acum
        if resp == "nao":
            yield from self._delegar(self._afirmar(s, r, o, True, 1.0), acum)
            return acum
        return ["Tudo bem, deixo de lado."]

    def _sugerir(self):
        b = self.ia.base
        firmes = [f for f in b.fatos.values() if f.fonte in FONTES_FIRMES]
        if len(firmes) < 15:
            return ["Ainda tenho poucos fatos para sugerir algo útil."]
        if len(firmes) > LIMITE_EMBEDDINGS:
            return [f"A base tem {len(firmes)} fatos informados; acima de {LIMITE_EMBEDDINGS} o treino "
                    "dos vetores passa do que cabe com folga no Colab."]
        impressao = hash(frozenset(f.chave for f in firmes))
        if self._emb_fp != impressao:
            epocas = max(40, min(200, 30000 // len(firmes)))
            self._emb = EmbeddingsTransE(dim=16, epocas=epocas).treinar(b)
            self._emb_fp = impressao
        uso = Counter(f.r for f in firmes if f.r not in ("é_um", "subclasse_de", "tem_propriedade",
                                                         "não_tem_propriedade"))
        relacoes = [r for r, n in uso.most_common(5) if n >= 3]
        if not relacoes:
            return ["Não tenho relações com exemplos suficientes para sugerir algo."]
        achados = [(p, t) for p, _, t in self._emb.sugerir(b, relacoes, top_k=6) if p >= 0.15][:3]
        if not achados:
            return ["Não encontrei nada que o padrão da base torne provável."]
        for p, t in achados:
            self.fila.append(self._q_sugestao(*t, p))
        return [f"Tenho {len(achados)} palpite(s) a partir do padrão da base. Vou perguntar."]

    def _caminho(self, a, c):
        b = self.ia.base
        conhecidas = {e for f in b.fatos.values() for e in (f.s, f.o)}
        for x in (a, c):
            if x not in conhecidas:
                return [f"Não conheço {_legivel(x)}."]
        if a == c:
            return ["São a mesma entidade."]
        g = nx.Graph()
        for f in b.fatos.values():
            if f.confianca >= LIMIAR_ACEITE and f.r not in ("tem_propriedade", "não_tem_propriedade"):
                if g.has_edge(f.s, f.o):
                    g[f.s][f.o]["fatos"].append(f)
                else:
                    g.add_edge(f.s, f.o, fatos=[f])
        try:
            caminho = nx.shortest_path(g, a, c)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [f"Não encontrei nenhuma ligação entre {_legivel(a)} e {_legivel(c)} na base."]
        if len(caminho) - 1 > 6:
            return [f"A ligação entre {_legivel(a)} e {_legivel(c)} tem mais de 6 passos."]
        linhas = [f"Ligação entre {_legivel(a)} e {_legivel(c)} em {len(caminho) - 1} passo(s):"]
        for n, (u, v) in enumerate(zip(caminho, caminho[1:]), 1):
            f = sorted(g[u][v]["fatos"], key=lambda x: (x.fonte not in FONTES_FIRMES, x.r))[0]
            marca = "" if f.fonte in FONTES_FIRMES else " (deduzido)"
            linhas.append(f"  {n}. {frase_fato(f.s, f.r, f.o)}{marca}")
        return linhas

    def _grafo(self, entidade=None):
        from matplotlib.figure import Figure
        b = self.ia.base
        g = nx.Graph()
        for f in b.fatos.values():
            if f.confianca < LIMIAR_ACEITE or f.r in ("tem_propriedade", "não_tem_propriedade"):
                continue
            if f.fonte not in FONTES_FIRMES:
                # deduções triviais (herança, inversa, simétrica) só poluem o desenho
                regra = self.ia.motor.regra(f.justificativa[0]) if f.justificativa else None
                if regra is None or regra.origem not in ("manual", "induzida"):
                    continue
            g.add_edge(f.s, f.o, relacao=f.r, deduzido=f.fonte not in FONTES_FIRMES)
        if g.number_of_nodes() == 0:
            return ["Ainda não há relações para desenhar."]
        if entidade:
            if entidade not in g:
                return [f"Não tenho relações sobre {_legivel(entidade)} para desenhar."]
            ego = nx.ego_graph(g, entidade, radius=2)
            ordem = [entidade] + [n for n in nx.bfs_tree(ego, entidade) if n != entidade]
            sub, titulo = ego.subgraph(ordem[:30]).copy(), entidade
        else:
            maiores = [n for n, _ in sorted(g.degree, key=lambda x: -x[1])[:40]]
            sub, titulo = g.subgraph(maiores).copy(), "visão geral"
        fig = Figure(figsize=(7.4, 4.8), dpi=130, facecolor="white")
        ax = fig.subplots()
        pos = nx.spring_layout(sub, seed=7, k=1.1 / max(1, sub.number_of_nodes()) ** 0.5)
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=520, edgecolors="#9aa3af", linewidths=0.8,
                               node_color=[MARCA["acento_claro"] if n == entidade else "#e9edf2" for n in sub])
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=6.5,
                                font_color="#111827")
        firmes = [(u, v) for u, v, d in sub.edges(data=True) if not d["deduzido"]]
        deduzidas = [(u, v) for u, v, d in sub.edges(data=True) if d["deduzido"]]
        nx.draw_networkx_edges(sub, pos, edgelist=firmes, ax=ax, edge_color="#6b7280", width=1.0)
        nx.draw_networkx_edges(sub, pos, edgelist=deduzidas, ax=ax, edge_color="#9aa3af",
                               style="dashed", width=0.9)
        rotulos = {(u, v): _legivel(d["relacao"]) for u, v, d in sub.edges(data=True)
                   if sub.number_of_edges() <= 25 or entidade in (u, v)}
        nx.draw_networkx_edge_labels(sub, pos, ax=ax, font_size=5.5, font_color="#6b7280",
                                     rotate=False, edge_labels=rotulos)
        ax.set_title(f"{MARCA['empresa']} | {MARCA['ia']}: {_legivel(titulo)}", fontsize=9,
                     color=MARCA["acento_claro"], loc="left")
        ax.axis("off")
        nome = "grafo_" + re.sub(r"\W+", "_", titulo) + ".png"
        fig.savefig(nome, bbox_inches="tight")
        return [f"[[img:{nome}]]",
                f"{sub.number_of_nodes()} entidades e {sub.number_of_edges()} relações. "
                "Linha tracejada indica relação deduzida por padrão. Deduções triviais "
                "(herança de classe) ficam fora do desenho."]

    def _descrever(self, e):
        b = self.ia.base
        fatos = [f for f in b.consultar(s=e, minimo=LIMIAR_ACEITE)]
        if not fatos:
            self._lembrar(e)
            return self._iniciar(self._q_classe(e, f"Não conheço {_legivel(e)}. "))
        self._lembrar(e)
        linhas = [f"Sobre {_legivel(e)}:"]
        for f in fatos[:14]:
            marca = "" if f.fonte in FONTES_FIRMES else f" (deduzido, {f.confianca:.0%})"
            linhas.append(f"  {frase_fato(f.s, f.r, f.o)}{marca}")
        if len(fatos) > 14:
            linhas.append(f"  e mais {len(fatos) - 14} fato(s).")
        membros = sorted(f.s for f in b.consultar(r="é_um", o=e, minimo=LIMIAR_ACEITE)
                         if f.fonte in FONTES_FIRMES)
        if membros:
            linhas.append("  Exemplos: " + ", ".join(membros[:8]))
        return linhas

    def _quem(self, r, o):
        fatos = self.ia.base.consultar(r=r, o=o, minimo=LIMIAR_ACEITE)
        if fatos:
            nomes = [_legivel(f.s) + ("" if f.fonte in FONTES_FIRMES else " (deduzido)")
                     for f in fatos]
            self._lembrar(o)
            return [f"Quem {SUPERFICIE.get(r, _legivel(r))} {_legivel(o)}: " + ", ".join(nomes) + "."]
        return self._iniciar(self._q_quem(r, o))

    def _o_que(self, s, r):
        fatos = self.ia.base.consultar(s=s, r=r, minimo=LIMIAR_ACEITE)
        if not fatos and r == "habitat_principal":
            alternativa = self.ia.base.consultar(s=s, r="vive_em", minimo=LIMIAR_ACEITE)
            if alternativa:
                r, fatos = "vive_em", alternativa
        if fatos:
            nomes = [_legivel(f.o) + ("" if f.fonte in FONTES_FIRMES else " (deduzido)")
                     for f in fatos]
            self._lembrar(s)
            return [f"{_legivel(s)} {SUPERFICIE.get(r, _legivel(r))}: " + ", ".join(nomes) + "."]
        return self._iniciar(self._q_o_que(s, r))

    def _porque_texto(self, f):
        if f.fonte in FONTES_FIRMES:
            arquivo = self._arquivo_de(f)
            if arquivo:
                p = next(p for p in f.procedencias if p.get("arquivo"))
                return f"isso consta em {arquivo}" + (f", linha {p['linha']}" if p.get("linha") else "")
            return "isso foi informado a mim na conversa"
        if not f.justificativa:
            return "é uma estimativa minha"
        regra = self.ia.motor.regra(f.justificativa[0])
        premissas = " e ".join(frase_fato(*p) for p in f.justificativa[1])
        origem = f" (regra '{regra.nome}', confiança {regra.confianca:.0%})" if regra else ""
        return f"{premissas}{origem}"

    def _por_que(self, s, r, o):
        self.ult_fato = (s, r, o)
        f = self.ia.base.fatos.get((s, r, o))
        if f is None or f.confianca < LIMIAR_ACEITE:
            return [f"Não tenho registrado que {frase_fato(s, r, o)}."]
        linhas = [f"{frase_fato(s, r, o)} porque {self._porque_texto(f)}."]
        if f.justificativa:
            for p in f.justificativa[1]:
                sub = self.ia.base.fatos.get(p)
                if sub is not None and sub.fonte not in FONTES_FIRMES:
                    linhas.append(f"  {frase_fato(*p)} porque {self._porque_texto(sub)}.")
        return linhas

    # --- subdiálogos de afirmação e verificação ---
    def _descartar(self, chave):
        f = self.ia.base.fatos.get(chave)
        self.registrar("descartado", frase_fato(*chave), chave=list(chave),
                       conf=round(f.confianca, 2) if f else None)
        self.ia.base.rejeitados.add(chave)
        self.ia.base.remover(chave)

    def _afirmar_varios(self, fatos, conf, ate=None):
        acum = []
        for s, r, o, negado in fatos:
            sub = self._afirmar(s, r, o, negado, conf, ate)
            yield from self._delegar(sub, acum)
        return acum

    def _afirmar(self, s, r, o, negado, conf, ate=None):
        b = self.ia.base
        s, r, o, negado = self.interprete._efetivo(s, r, o, negado)
        chave = (s, r, o)
        if self.grau < 1.0:
            if negado:
                # "provavelmente não": enfraquece o fato, sem descartá-lo
                existia = b.fatos.get(chave)
                if existia is None:
                    return ["Anotei sua dúvida, mas não tenho esse fato para enfraquecer."]
                existia.enfraquecer(4.0)
                self.mudou = True
                self.ia.pensar(verbose=False)
                return [f"Anotei como provável que não: {frase_fato(*chave)} agora tem confiança "
                        f"{existia.confianca:.0%}."]
            conf = min(conf, self.grau)
        if negado:
            # negação de uma relação sem oposta declarada: guarda como fato descartado
            existia = b.fatos.get(chave)
            self._descartar(chave)
            self.mudou = True
            if existia is not None and existia.confianca >= LIMIAR_ACEITE:
                return [f"Entendi, então {frase_fato(*chave)} não é verdade. Tirei isso da base."]
            return [f"Anotei que não é verdade que {frase_fato(*chave)}."]
        existente = b.fatos.get(chave)
        oposta = b.relacao_oposta(r)
        contra = b.fatos.get((s, oposta, o)) if oposta else None
        nota = []
        if existente is not None and existente.confianca >= LIMIAR_ACEITE:
            era_deduzido = existente.fonte not in FONTES_FIRMES
            b.adicionar(s, r, o, conf, "observado", valido_ate=ate)
            self.mudou = True
            if era_deduzido:
                return [f"Eu já tinha deduzido que {frase_fato(*chave)}. Agora você confirmou."]
            return [f"Isso confere com o que eu tinha. Reforcei a confiança em "
                    f"{b.fatos[chave].confianca:.0%}."]
        if contra is not None and contra.confianca >= LIMIAR_ACEITE:
            if contra.fonte in FONTES_FIRMES:
                resp = yield (f"Eu tinha registrado que {frase_fato(*contra.chave)} "
                              f"(confiança {contra.confianca:.0%}, {ROTULO_FONTE[contra.fonte]}). "
                              f"Você está dizendo que {frase_fato(*chave)}. Tem certeza?", "sim_nao")
                if resp != "sim":
                    return ["Mantive o que eu tinha registrado."]
                self._descartar(contra.chave)
            else:
                nota.append(f"Isso contraria o que eu deduzia ({frase_fato(*contra.chave)}). "
                            "Passo a tratar como exceção.")
        if r in b.funcionais:
            outros = [f for f in b.consultar(s=s, r=r, minimo=LIMIAR_ACEITE)
                      if f.o != o and f.fonte in FONTES_FIRMES]
            if outros:
                antigo = outros[0]
                resp = yield (f"Eu tinha que {frase_fato(*antigo.chave)} "
                              f"(confiança {antigo.confianca:.0%}) e só guardo um valor para "
                              f"'{_legivel(r)}'. Troco por {frase_fato(*chave)}?", "sim_nao")
                if resp != "sim":
                    return ["Mantive o valor anterior."]
                self._descartar(antigo.chave)
        if r == "é_um":
            for a, c in b.disjuntas:
                if o not in (a, c):
                    continue
                outra = c if o == a else a
                rival = b.fatos.get((s, "é_um", outra))
                if rival is not None and rival.confianca >= LIMIAR_ACEITE and rival.fonte in FONTES_FIRMES:
                    resp = yield (f"Eu tinha {_legivel(s)} como {outra}, e {a} e {c} não podem "
                                  f"ser a mesma coisa. Troco para {o}?", "sim_nao")
                    if resp != "sim":
                        return ["Mantive o que eu tinha registrado."]
                    self._descartar(rival.chave)
        antes = {k for k, f in b.fatos.items()
                 if f.fonte == "inferido" and f.confianca >= LIMIAR_ACEITE}
        conhecido = s in {f.s for f in b.fatos.values()} | {f.o for f in b.fatos.values()}
        b.adicionar(s, r, o, conf, "observado", valido_ate=ate)
        self.registrar("anotado", frase_fato(*chave))
        self.ia.pensar(verbose=False)
        self.mudou = True
        self.aprendidos += 1
        depois = {k for k, f in b.fatos.items()
                  if f.fonte == "inferido" and f.confianca >= LIMIAR_ACEITE}
        novos = sorted(depois - antes)
        destaque = [k for k in novos if k[1] != "é_um"] or novos
        texto = f"Anotei: {frase_fato(*chave)}."
        if ate:
            texto = f"Anotei: {frase_fato(*chave)}, válido até {ate[8:]}/{ate[5:7]}/{ate[:4]}."
        if conf < 1.0:
            texto = f"Anotei com confiança {conf:.0%}: {frase_fato(*chave)}."
        linhas = [texto] + nota
        if not conhecido:
            linhas.append(f"{_legivel(s)} é novo para mim.")
        if destaque:
            linhas.append("Com isso, deduzi: " + "; ".join(frase_fato(*k) for k in destaque[:3])
                          + ("." if len(destaque) <= 3 else f"; e mais {len(destaque) - 3}."))
        return linhas

    def _nova_relacao(self, s, nome, o, negado, conf):
        resp = yield (f"Não conheço '{_legivel(nome)}' como relação. Guardo como uma "
                      f"relação nova entre {_legivel(s)} e {_legivel(o)}?", "sim_nao")
        if resp != "sim":
            return ["Certo, não guardei."]
        self._lembrar(s)
        acum = []
        yield from self._delegar(self._afirmar(s, nome, o, negado, conf), acum)
        for r in self.recursos:
            yield from self._delegar(r.apos_nova_relacao(s, nome, o), acum)
        return acum

    def _verificar(self, s, r, o, negado):
        b = self.ia.base
        chave = (s, r, o)
        f = b.fatos.get(chave)
        oposta = b.relacao_oposta(r)
        contra = b.fatos.get((s, oposta, o)) if oposta else None
        if f is not None and f.confianca >= LIMIAR_ACEITE:
            self.ult_fato = chave
            quem = ROTULO_FONTE[f.fonte]
            abre = ("Sim." if f.confianca >= 0.9 else "Provavelmente sim." if f.confianca >= 0.7
                    else "Talvez.")
            texto = (f"{abre} {frase_fato(*chave)} (confiança {self._conf_txt(f.confianca)}, {quem}).")
            if f.fonte not in FONTES_FIRMES:
                texto += f" Deduzi isso porque {self._porque_texto(f)}."
            return [texto] if not negado else [f"Não. Tenho registrado que {frase_fato(*chave)}."]
        if contra is not None and contra.confianca >= LIMIAR_ACEITE:
            return [f"Não. Tenho registrado que {frase_fato(*contra.chave)} "
                    f"(confiança {contra.confianca:.0%}, {ROTULO_FONTE[contra.fonte]})."]
        if chave in b.rejeitados:
            return ["Não. Isso já foi descartado antes."]
        trans = self.rec.get("transparencia")
        venc = trans._vencedor(chave) if trans is not None else None
        if venc is not None and venc.confianca >= LIMIAR_ACEITE:
            return [f"Não. Tenho {frase_fato(*venc.chave)} ({self._conf_txt(venc.confianca)}), "
                    "que não combina com isso. Pergunte por que não, se quiser ver o motivo."]
        resp = yield (f"Não tenho essa informação. Você sabe se {frase_fato(*chave)}?", "sim_nao")
        acum = []
        if resp == "sim":
            yield from self._delegar(self._afirmar(s, r, o, False, 1.0), acum)
            return acum
        if resp == "nao":
            yield from self._delegar(self._afirmar(s, r, o, True, 1.0), acum)
            return acum
        for _, _, faltante, _ in self.ia.motor.abduzir(s, r, o, limite=2):
            resp = yield (f"Então me ajude por outro caminho: {frase_fato(*faltante)}?", "sim_nao")
            if resp == "sim":
                yield from self._delegar(self._afirmar(*faltante, False, 1.0), acum)
                fato = b.fatos.get(chave)
                if fato is not None and fato.confianca >= LIMIAR_ACEITE:
                    acum.append(f"Com isso concluo que {frase_fato(*chave)} "
                                f"(confiança {fato.confianca:.0%}).")
                return acum
            if resp == "nao":
                yield from self._delegar(self._afirmar(*faltante, True, 1.0), acum)
        return acum + ["Sem essa informação, não consigo concluir."]

    # --- perguntas que a IA faz por iniciativa própria ---
    def _fatos_da_resposta(self, resposta, variantes):
        for tentativa in [resposta] + variantes:
            analise = self.interprete.afirmativa(tentativa)
            if analise and "fatos" in analise:
                return analise["fatos"], analise["conf"]
        return [], 1.0

    def _tratar_resposta(self, resposta, variantes):
        """Subdiálogo: interpreta uma resposta livre e grava o que ela afirma."""
        tipo = classificar_resposta(resposta)
        if tipo == "sim":
            resposta = yield ("Pode contar.", "livre")
            tipo = classificar_resposta(resposta)
        if tipo in ("nao", "nao_sei"):
            return ["Tudo bem, deixo isso para depois."]
        fatos, conf = self._fatos_da_resposta(resposta, variantes)
        if not fatos:
            return ["Não consegui interpretar a resposta. Pode dizer de outro jeito, "
                    "por exemplo: o gato caça o rato."]
        self._lembrar(*[f[0] for f in fatos])
        acum = []
        yield from self._delegar(self._afirmar_varios(fatos, conf), acum)
        return acum

    def _nome_solto(self, texto):
        """O nome de uma entidade já conhecida, se `texto` não for mais que isso (com ou sem artigo)."""
        itp = self.interprete
        palavras = itp._sem_artigos(itp.tokens(texto))
        if not palavras or len(palavras) > 3 or "," in palavras:
            return None
        chave = _chave(" ".join(palavras))
        if chave in itp.propriedades():
            return None
        return itp.entidades().get(chave)

    def _q_classe(self, e, prefixo=""):
        resp = yield (f"{prefixo}O que é {_legivel(e)}? Diga a classe, por exemplo Mamifero, "
                      "ou conte o que souber.", "livre")
        return (yield from self._tratar_resposta(resp, [f"{e} é {resp}", f"{e} {resp}"]))

    def _q_aberta(self, e, e_classe=False):
        texto = (f"O que você sabe que vale para qualquer item da classe {_legivel(e)}? "
                 "Por exemplo, uma característica ou um hábito." if e_classe
                 else f"O que mais você sabe sobre {_legivel(e)}?")
        resp = yield (texto, "livre")
        nome = self._nome_solto(resp)
        if nome is not None:
            # só o nome de um assunto conhecido: é um pedido para falar dele, não uma característica de `e`
            return self._descrever(nome)
        return (yield from self._tratar_resposta(resp, [f"{e} {resp}", f"{e} tem {resp}"]))

    def _q_valor(self, e, r):
        modelo = PERGUNTA_VALOR.get(r, "Qual é o valor de '" + _legivel(r) + "' para {e}?")
        resp = yield (modelo.format(e=_legivel(e)), "livre")
        sup = SUPERFICIE.get(r, _legivel(r))
        return (yield from self._tratar_resposta(resp, [f"{e} {sup} {resp}"]))

    def _q_quem(self, r, o):
        sup = SUPERFICIE.get(r, _legivel(r))
        resp = yield (f"Não tenho registro de quem {sup} {_legivel(o)}. Você sabe?", "livre")
        return (yield from self._tratar_resposta(resp, [f"{resp} {sup} {o}"]))

    def _q_o_que(self, s, r):
        sup = SUPERFICIE.get(r, _legivel(r))
        pergunta = (f"Não sei onde {_legivel(s)} vive. Você sabe?" if r == "habitat_principal"
                    else f"Não tenho registro do que {_legivel(s)} {sup}. Você sabe?")
        resp = yield (pergunta, "livre")
        return (yield from self._tratar_resposta(resp, [f"{s} {sup} {resp}"]))

    def _q_lacuna(self, e, classe, r, o, n):
        sim_nao = frase_fato(e, r, o)
        resp = yield (f"{sim_nao}? Entre os outros da classe {classe}, {n} têm essa "
                      "característica.", "sim_nao")
        if resp == "sim":
            acum = []
            yield from self._delegar(self._afirmar(e, r, o, False, 1.0), acum)
            return acum
        if resp == "nao":
            acum = []
            yield from self._delegar(self._afirmar(e, r, o, True, 1.0), acum)
            return acum
        return ["Tudo bem."]

    def _q_verificar(self, fato):
        chave = fato.chave
        resp = yield (f"Deduzi que {frase_fato(*chave)}, porque {self._porque_texto(fato)}. "
                      "Isso procede?", "sim_nao")
        if resp in ("sim", "nao") and self.grau < 1.0:
            self.registrar("verificacao", frase_fato(*chave), aceita=(resp == "sim"), grau=self.grau)
            self.ia.feedback(*chave, correto=(resp == "sim"), imprimir=False, peso=self.grau)
            self.mudou = True
            return ["Anotei como provável. Ajustei a confiança só um pouco, sem confirmar nem descartar."]
        if resp == "sim":
            self._rotular(chave, True)
            self.registrar("verificacao", frase_fato(*chave), aceita=True)
            self.ia.feedback(*chave, correto=True, imprimir=False)
            self.mudou = True
            return ["Certo, confirmei essa dedução."]
        if resp == "nao":
            self._rotular(chave, False)
            self.registrar("verificacao", frase_fato(*chave), aceita=False)
            self.ia.feedback(*chave, correto=False, imprimir=False)
            self.mudou = True
            self.aprendidos += 1
            return ["Entendi. Registrei como exceção e não vou mais concluir isso."]
        return ["Tudo bem, mantenho como dedução."]

    def _q_conflito(self, a, b):
        base = self.ia.base
        fa, fb = base.fatos.get(a), base.fatos.get(b)
        if fa is None or fb is None:
            return []
        manter, descartar = (fa, fb) if fa.confianca >= fb.confianca else (fb, fa)
        resp = yield (f"Tenho duas informações que não podem valer juntas: "
                      f"{frase_fato(*manter.chave)} (confiança {manter.confianca:.0%}) e "
                      f"{frase_fato(*descartar.chave)} (confiança {descartar.confianca:.0%}). "
                      f"Mantenho a primeira?", "sim_nao")
        if resp == "sim":
            vencedor, perdedor = manter, descartar
        elif resp == "nao":
            vencedor, perdedor = descartar, manter
        else:
            return ["Deixo as duas por enquanto."]
        self._descartar(perdedor.chave)
        self.registrar("decisao", f"Conflito: mantive {frase_fato(*vencedor.chave)}; "
                                  f"descartei {frase_fato(*perdedor.chave)}.")
        saida = ["Certo, ajustei a base."]
        av, ap = self._arquivo_de(vencedor), self._arquivo_de(perdedor)
        if av != ap:
            base.registrar_fonte(av, True)
            base.registrar_fonte(ap, False)
            if ap:
                saida.append(f"Passo a confiar um pouco menos em '{ap}' "
                             f"(peso {base.peso_fonte(ap):.2f} nos próximos fatos que ele trouxer).")
        self.ia.pensar(verbose=False)
        self.mudou = True
        return saida

    # --- escolha da próxima pergunta ---
    def _ordenar(self, entidades, foco):
        ordem = {e: i for i, e in enumerate(self.recentes)}
        escolhidas = [e for e in entidades if foco is None or e in foco]
        return sorted(escolhidas, key=lambda e: (ordem.get(e, 999), e))

    def _buscar_classe(self, foco):
        b = self.ia.base
        sujeitos = {f.s for f in b.fatos.values() if f.fonte in FONTES_FIRMES}
        classes = ({f.o for f in b.fatos.values() if f.r in ("é_um", "subclasse_de")}
                   | {f.s for f in b.fatos.values() if f.r == "subclasse_de"})
        com_classe = {f.s for f in b.fatos.values() if f.r == "é_um"}
        for e in self._ordenar(sujeitos, foco):
            chave = ("classe", e)
            if e not in classes and e not in com_classe and chave not in self.perguntadas:
                return chave, self._q_classe(e)
        relacionais = {"predador_de", "presa_de", "compete_com", "parceiro_de", "evoluiu_de"}
        objetos = {f.o for f in b.fatos.values() if f.r in relacionais}
        for e in self._ordenar(objetos, foco):
            chave = ("classe", e)
            if (e not in classes and e not in com_classe and e not in sujeitos
                    and chave not in self.perguntadas):
                return chave, self._q_classe(e)
        return None

    def _buscar_classe_nova(self, foco):
        b = self.ia.base
        com_dados = {f.s for f in b.fatos.values()}
        for e in self._ordenar({f.o for f in b.fatos.values() if f.r == "é_um"
                                and f.fonte in FONTES_FIRMES}, foco):
            chave = ("classe_nova", e)
            if e not in com_dados and chave not in self.perguntadas:
                return chave, self._q_aberta(e, e_classe=True)
        return None

    def _buscar_verificacao(self, foco, regras):
        candidatos = []
        for f in self.ia.base.fatos.values():
            if f.fonte != "inferido" or f.confianca < LIMIAR_ACEITE or not f.justificativa:
                continue
            regra = self.ia.motor.regra(f.justificativa[0])
            if regra is None or not regras(regra) or (foco is not None and f.s not in foco):
                continue
            if (regra.nome == "heranca_de_propriedade"
                    and self.ia._profundidade_classe(f.justificativa[1][0][2]) <= 2):
                continue  # herdar "respira" de Animal é trivial demais para perguntar
            chave = ("verificar", f.chave)
            if regra.nome == "heranca_de_propriedade" and foco is None:
                # varredura geral: uma vez por par (classe, propriedade), senão a IA
                # repetiria a mesma dúvida para cada animal da classe
                chave = ("verificar_heranca", f.justificativa[1][0][2], f.o)
            if chave not in self.perguntadas:
                candidatos.append((f.confianca, chave, f))
        if not candidatos:
            return None
        _, chave, fato = min(candidatos, key=lambda c: (c[0], c[1]))
        self._meta[chave] = fato
        return chave, self._q_verificar(fato)

    def _buscar_lacuna(self, foco):
        b = self.ia.base
        membros = defaultdict(list)
        for f in b.fatos.values():
            if f.r == "é_um" and f.fonte in FONTES_FIRMES:
                membros[f.o].append(f.s)
        alvos = {m for ms in membros.values() for m in ms}
        for e in self._ordenar(alvos, foco):
            for classe in sorted(c for c, ms in membros.items() if e in ms):
                pares = [p for p in membros[classe] if p != e]
                contagem = Counter()
                for p in pares:
                    for f in b.consultar(s=p, minimo=LIMIAR_ACEITE):
                        if f.fonte in FONTES_FIRMES and (f.r == "tem_propriedade" or f.r in b.funcionais):
                            contagem[(f.r, f.o if f.r == "tem_propriedade" else None)] += 1
                for (r, o), n in sorted(contagem.items(), key=lambda x: (-x[1], str(x[0]))):
                    if r == "tem_propriedade":
                        chave = ("lacuna", e, r, o)
                        conhecido = (b.fatos.get((e, r, o)) or b.fatos.get((e, "não_tem_propriedade", o)))
                        if (chave in self.perguntadas or (e, r, o) in b.rejeitados
                                or (conhecido is not None and conhecido.confianca >= LIMIAR_ACEITE)):
                            continue
                        self._meta[chave] = (n, len(pares))
                        return chave, self._q_lacuna(e, classe, r, o, n)
                    chave = ("valor", e, r)
                    if chave in self.perguntadas or b.consultar(s=e, r=r, minimo=LIMIAR_ACEITE):
                        continue
                    return chave, self._q_valor(e, r)
        return None

    def _buscar_aberta(self, foco):
        b = self.ia.base
        contagem = Counter(f.s for f in b.fatos.values()
                           if f.fonte in FONTES_FIRMES and f.r != "é_um")
        for e in self._ordenar(set(contagem) | set(self.recentes), foco):
            chave = ("aberta", e)
            if contagem.get(e, 0) < 3 and chave not in self.perguntadas:
                return chave, self._q_aberta(e)
        return None

    @staticmethod
    def _entropia(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def _pontuar(self, chave, dependentes):
        """Valor esperado de perguntar: incerteza da resposta somada ao impacto dela na base."""
        tipo = chave[0]
        nome = _legivel(str(chave[1])) if len(chave) > 1 and isinstance(chave[1], str) else ""
        if tipo == "classe":
            return 1.0, f"eu ainda não sei a que classe {nome} pertence"
        if tipo == "classe_nova":
            return 0.95, f"você citou {nome} como classe, mas eu não tenho nenhum fato sobre ela"
        if tipo in ("verificar", "verificar_heranca"):
            fato = self._meta.get(chave)
            p = fato.confianca if fato is not None else 0.6
            if tipo == "verificar":
                n = dependentes.get(chave[1], 0)
                return (0.4 + 0.8 * self._entropia(p) + 0.1 * min(n, 5),
                        f"deduzi isso com {p:.0%} de confiança"
                        + (f" e {n} outra(s) conclusão(ões) dependem disso" if n else ""))
            return (0.2 + 0.5 * self._entropia(p),
                    f"deduzi isso por herança de classe, com {p:.0%} de confiança")
        if tipo == "lacuna":
            n, total = self._meta.get(chave, (1, 2))
            return (0.3 + 0.7 * self._entropia(n / max(total, 1)),
                    f"{n} de {total} outros da mesma classe têm essa característica, e {nome} não está registrado")
        if tipo == "valor":
            return 0.35, f"outros da mesma classe de {nome} têm esse dado, e ele ainda não tem"
        if tipo == "aberta":
            return 0.15, f"tenho poucos fatos sobre {nome}"
        return 0.1, "queria completar a base"

    def _proxima_pergunta(self):
        """Escolhe, entre todas as perguntas possíveis, a de maior valor. O que a pessoa
        acabou de mencionar ganha um reforço, para a conversa não saltar de assunto."""
        if self.recentes:
            niveis = [(self.recentes[:1], 1.6), (self.recentes[:3], 1.3), (None, 1.0)]
        else:
            niveis = [(None, 1.0)]
        dependentes = Counter(p for f in self.ia.base.fatos.values() if f.justificativa
                              for p in f.justificativa[1])
        finders = (self._buscar_classe, self._buscar_classe_nova,
                   lambda f: self._buscar_verificacao(f, lambda r: r.origem in ("manual", "induzida")),
                   self._buscar_lacuna,
                   lambda f: self._buscar_verificacao(f, lambda r: r.nome == "heranca_de_propriedade"),
                   self._buscar_aberta)
        melhor = None
        for foco, reforco in niveis:
            candidatas = []
            for buscar in finders:
                achado = buscar(foco)
                if achado:
                    score, motivo = self._pontuar(achado[0], dependentes)
                    candidatas.append((score, achado[0], achado[1], motivo))
            for r in self.recursos:
                for c in r.perguntas(foco):
                    if c["chave"] in self.perguntadas:
                        continue
                    if foco is not None and c.get("entidade") not in foco:
                        continue
                    candidatas.append((c["score"], c["chave"], c["gerador"], c["motivo"]))
            for score, chave, gerador, motivo in candidatas:
                total = score * reforco
                if melhor is None or total > melhor[0]:
                    melhor = (total, chave, gerador, motivo)
        return melhor and melhor[1:]

    def _iniciativa(self):
        if not self.curioso or self.ativo is not None:
            return []
        if self.fila:
            self.ultimo_motivo = "tenho uma pendência anotada para resolver com você"
            return self._iniciar(self.fila.pop(0))
        if not self.ia.base.fatos:
            return []
        achado = self._proxima_pergunta()
        if achado is None:
            if self._avisou_fim:
                return []
            self._avisou_fim = True
            return ["Não tenho mais perguntas por enquanto. Me conte o que mais você souber."]
        chave, gerador, motivo = achado
        self.perguntadas.add(chave)
        self.ultimo_motivo = motivo
        self.registrar("pergunta", motivo)
        return self._iniciar(gerador)


# %% [22] Recursos: procedência dos fatos e confiabilidade das fontes
# Cada fato guarda de onde veio (arquivo, linha, conversa, momento). Quando duas fontes se
# contradizem e a pessoa decide quem tem razão, a fonte vencedora e a perdedora recebem um
# ponto. A confiabilidade de uma fonte é a média de uma Beta com prior (19, 1), dividida pela
# média do prior (0,95) e limitada a 1: uma fonte sem histórico pesa 1,0 e cada derrota a
# faz pesar um pouco menos nos fatos que ela trouxer em novas leituras.

def _nome_rel(texto):
    """Cabeçalho de planilha ou nome de campo vira nome de relação: sem acento, com sublinhado."""
    return re.sub(r"\W+", "_", _sem_acento(str(texto))).strip("_")


def _valor(texto):
    """Valor de uma célula vira nome de entidade: espaços viram sublinhado, o resto fica."""
    return re.sub(r"\s+", "_", str(texto).strip())


def _data_br(iso):
    return f"{iso[8:10]}/{iso[5:7]}/{iso[:4]}" if iso else ""


def _momento(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return str(iso)


@recurso
class Fontes(Recurso):
    nome = "fontes"
    tipos = ("procedencia", "fontes")

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:quais )?(?:sao )?(?:as )?(?:fontes|confiabilidade das fontes|"
                    r"fontes de dados|arquivos que voce leu|quais arquivos voce leu)$", juntas):
            return {"tipo": "fontes"}
        m = re.match(r"^(?:de onde (?:veio|vem|surgiu)|qual (?:e )?(?:a )?(?:fonte|origem) "
                     r"(?:de|do|da|dos|das)|onde voce (?:viu|leu|aprendeu))\s+(.+)$", juntas)
        if m:
            analise = itp.afirmativa(m.group(1))
            if analise and "fatos" in analise:
                s, r, o, neg = analise["fatos"][0]
                return {"tipo": "procedencia", "fato": itp._efetivo(s, r, o, neg)[:3]}
            return {"tipo": "nada"}
        return None

    def tratar(self, intencao):
        if intencao["tipo"] == "fontes":
            return self._resumo_fontes()
        chave = intencao["fato"]
        f = self.base.fatos.get(chave)
        if f is None or f.confianca < LIMIAR_ACEITE:
            return [f"Não tenho registrado que {frase_fato(*chave)}."]
        if f.fonte not in FONTES_FIRMES:
            return self.d._por_que(*chave)
        if not f.procedencias:
            return [f"{frase_fato(*chave)}: não guardei a origem deste fato."]
        linhas = [f"{frase_fato(*chave)}. Origem:"]
        for p in f.procedencias:
            quando = _momento(p.get("quando"))
            if p.get("arquivo"):
                linhas.append(f"  arquivo {p['arquivo']}"
                              + (f", linha {p['linha']}" if p.get("linha") else "") + f", lido em {quando}")
            else:
                linhas.append(f"  dito na conversa, em {quando}")
        if f.valido_ate:
            linhas.append(f"  vale até {_data_br(f.valido_ate)}")
        return linhas

    def _resumo_fontes(self):
        b = self.base
        contagem = Counter()
        for f in b.fatos.values():
            if f.fonte in FONTES_FIRMES:
                contagem[self.d._arquivo_de(f) or "conversa"] += 1
        if not contagem:
            return ["Ainda não tenho fatos informados, então não tenho fontes para listar."]
        linhas = ["Fontes dos fatos informados:"]
        for fonte, n in contagem.most_common(12):
            ac, er = b.fontes_stats.get(fonte, (0, 0))
            peso = b.peso_fonte(fonte)
            extra = f", {ac} disputa(s) ganha(s) e {er} perdida(s), peso {peso:.2f}" if ac or er else ""
            linhas.append(f"  {fonte}: {n} fato(s){extra}")
        if len(contagem) > 12:
            linhas.append(f"  e mais {len(contagem) - 12} fonte(s).")
        linhas.append("O peso cai quando uma fonte perde uma disputa que você decidiu.")
        return linhas


# %% [23] Recursos: tabelas largas, campos obrigatórios e modelos de domínio
# Planilha larga = uma linha por coisa e uma coluna por atributo (o formato de um inventário
# de patrimônio). Cada célula vira um fato: linha de patrimônio 123, coluna setor, valor TI
# vira (123, setor, TI). O Info adivinha qual coluna identifica a linha e qual traz a classe,
# e pergunta se entendeu certo antes de seguir.

CAMPOS_CLASSE = ("classe", "tipo", "categoria", "class", "type", "tipo_de_ativo", "tipo_do_item")
CAMPOS_CHAVE = ("patrimonio", "tombo", "tombamento", "id", "codigo", "cod", "identificador",
                "numero_de_serie", "serie", "matricula", "entidade", "nome", "numero_do_processo",
                "processo", "numero")


def _classe_de(texto):
    """Valor da coluna de classe vira nome de classe: sem acento e em CamelCase."""
    palavras = re.split(r"[\s_\-]+", _sem_marcas(str(texto).strip()))
    return "".join(p[:1].upper() + p[1:] for p in palavras if p)


def detectar_tabela_larga(linhas, ia):
    """Devolve a descrição da tabela (cabecalho, linhas, coluna chave e de classe) ou None
    quando as linhas parecem triplas (sujeito, relação, objeto)."""
    if len(linhas) < 3 or len(linhas[0]) < 2:
        return None
    cab = [_nome_rel(c) for c in linhas[0]]
    if not all(cab) or len(set(cab)) != len(cab):
        return None
    corpo = [l for l in linhas[1:] if any(c.strip() for c in l)]
    if len(corpo) < 2:
        return None
    conhecidas = {_chave(r) for r in FORMAS_RELACAO} | {_chave(f.r) for f in ia.base.fatos.values()}
    conhecidas |= {_chave(r) for r in ("é_um", "tem_propriedade", "subclasse_de", "caça")}
    if len(cab) >= 3 and sum(_chave(l[1]) in conhecidas for l in corpo if len(l) > 1) >= len(corpo) / 2:
        return None  # a segunda coluna é de relações: são triplas sem cabeçalho de nomes
    n = len(corpo)

    def preenchida(i):
        return sum(1 for l in corpo if i < len(l) and l[i].strip())

    def unica(i):
        valores = [l[i].strip() for l in corpo if i < len(l) and l[i].strip()]
        return len(valores) >= 0.95 * n and len(set(valores)) >= 0.95 * len(valores)
    candidatas = [i for i in range(len(cab)) if unica(i)]
    por_nome = [i for i in candidatas if cab[i] in CAMPOS_CHAVE]
    chave = (por_nome or candidatas or [0])[0]
    classe = next((i for i, h in enumerate(cab) if h in CAMPOS_CLASSE and i != chave
                   and 0 < len({l[i].strip() for l in corpo if i < len(l) and l[i].strip()}) <= max(2, n // 2)
                   and preenchida(i) >= 0.8 * n), None)
    return {"cabecalho": cab, "originais": [c.strip() for c in linhas[0]], "linhas": linhas[1:],
            "chave": chave, "classe": classe, "classe_fixa": None, "novos": [], "arquivo": None}


def importar_tabela(ia, meta):
    """Grava os fatos da tabela e guarda quais são novos (para desfazer, se a pessoa discordar)."""
    b = ia.base
    anterior = b.proc_atual
    arquivo = meta.get("arquivo") or anterior.get("arquivo")
    kc, cc = meta["chave"], meta["classe"]
    chaves_tabela = b.config.setdefault("chaves", [])
    if meta["cabecalho"][kc] not in chaves_tabela:
        chaves_tabela.append(meta["cabecalho"][kc])  # a coluna chave já é o campo de mesmo nome
    lidas = 0
    try:
        for numero, linha in enumerate(meta["linhas"], 2):
            if kc >= len(linha) or not linha[kc].strip():
                continue
            entidade = _valor(linha[kc])
            b.proc_atual = {"via": "arquivo", "arquivo": arquivo, "linha": numero}
            pares = []
            if meta["classe_fixa"]:
                pares.append(("é_um", meta["classe_fixa"]))
            for i, h in enumerate(meta["cabecalho"]):
                if i == kc or i >= len(linha) or not linha[i].strip():
                    continue
                if i == cc:
                    pares.append(("é_um", _classe_de(linha[i])))
                else:
                    pares += [(h, _valor(v)) for v in linha[i].split(";") if v.strip()]
            for r, o in pares:
                chave = (entidade, b.normalizar_relacao(r), o)
                if chave not in b.fatos or b.fatos[chave].fonte not in FONTES_FIRMES:
                    meta["novos"].append(chave)
                b.adicionar(entidade, r, o, 1.0, "observado")
                lidas += 1
    finally:
        b.proc_atual = anterior
    return lidas


def exportar_tabela(ia, caminho, classe=None):
    """Planilha larga: uma linha por entidade, uma coluna por relação. Só fatos informados."""
    b = ia.base
    firmes = [f for f in b.fatos.values() if f.fonte in FONTES_FIRMES]
    classes = defaultdict(list)
    for f in firmes:
        if f.r == "é_um":
            classes[f.s].append(f.o)
    if classe:
        entidades = sorted(e for e, cs in classes.items() if classe in cs)
    else:
        entidades = sorted({f.s for f in firmes})
    relacoes = sorted({f.r for f in firmes if f.s in set(entidades) and f.r != "é_um"})
    valores = defaultdict(list)
    for f in firmes:
        if f.r != "é_um":
            valores[(f.s, f.r)].append(_legivel(f.o))
    with open(caminho, "w", encoding="utf-8-sig", newline="") as fp:
        escritor = csv.writer(fp)
        chaves = ia.base.config.get("chaves", [])
        escritor.writerow([chaves[0] if len(chaves) == 1 else "entidade", "classe"] + relacoes)
        for e in entidades:
            escritor.writerow([_legivel(e), ";".join(_legivel(c) for c in classes.get(e, []))]
                              + [";".join(valores.get((e, r), [])) for r in relacoes])
    return len(entidades), relacoes


@recurso
class TabelaLarga(Recurso):
    nome = "tabela"
    tipos = ("exportar_tabela",)

    def interpretar(self, itp, bruto, juntas):
        m = re.match(r"^export\w*\s+(?:a\s+|as\s+)?(?:tabela|planilha larga|inventario)"
                     r"(?:\s+(?:de|dos|das|do|da)\s+(.+))?$", juntas)
        if m:
            classe = itp.nome_entidade(m.group(1).split()) if m.group(1) else None
            return {"tipo": "exportar_tabela", "classe": classe}
        return None

    def tratar(self, intencao):
        classe = intencao["classe"]
        caminho = f"{self.d._stem()}_tabela{'_' + classe.lower() if classe else ''}.csv"
        try:
            n, rels = exportar_tabela(self.ia, caminho, classe)
        except OSError as erro:
            return [f"Não consegui gravar a tabela: {erro}"]
        if not n:
            return ["Não há entidades para a tabela" + (f" da classe {classe}." if classe else ".")]
        return [f"Exportei {n} linha(s) e {len(rels)} coluna(s) de atributos em '{caminho}'.",
                f"[[baixar:{caminho}]]"]

    def ao_carregar(self, info):
        for meta in info.get("tabelas", []):
            self.d.fila.append(self._q_tabela(meta))
        return []

    def _desfazer(self, meta):
        for chave in meta["novos"]:
            self.base.remover(chave)
        meta["novos"] = []

    def _q_tabela(self, meta):
        b = self.base
        cab = meta["originais"]
        chave, cc = cab[meta["chave"]], meta["classe"]
        atributos = [c for i, c in enumerate(cab) if i not in (meta["chave"], cc)]
        linhas = len([l for l in meta["linhas"] if any(c.strip() for c in l)])
        classe_txt = (f"o tipo vem da coluna '{cab[cc]}'" if cc is not None
                      else "não encontrei coluna de tipo")
        texto = (f"Li '{meta['arquivo']}' como uma tabela com {linhas} linha(s). Cada linha é uma "
                 f"coisa identificada pela coluna '{chave}'; {classe_txt}. Atributos: "
                 + ", ".join(atributos[:8]) + (", ..." if len(atributos) > 8 else "")
                 + ". Interpretei certo?")
        resp = yield (texto, "sim_nao")
        if resp == "nao":
            self._desfazer(meta)
            resp = yield ("Qual coluna identifica cada linha? Escreva o nome dela "
                          "(por exemplo: patrimônio).", "livre")
            nomes = [_nome_rel(c) for c in cab]
            alvo = _nome_rel(resp)
            if alvo not in nomes:
                return [f"Não achei a coluna '{resp}'. Tirei a tabela da base; "
                        "carregue de novo depois de conferir os cabeçalhos."]
            meta["chave"] = nomes.index(alvo)
            if meta["classe"] == meta["chave"]:
                meta["classe"] = None
            importar_tabela(self.ia, meta)
            self.ia.pensar(verbose=False)
            self.d.mudou = True
        acum = ["Certo, mantenho essa leitura."] if resp != "nao" else []
        if meta["classe"] is None and not meta["classe_fixa"]:
            resp = yield ("Cada linha da tabela é um quê? Diga a classe, por exemplo Notebook.", "livre")
            if classificar_resposta(resp) in ("nao", "nao_sei"):
                return acum + ["Tudo bem, deixo sem classe por enquanto."]
            classe = self.d.interprete.nome_entidade(resp.split())
            for linha in meta["linhas"]:
                ent = _valor(linha[meta["chave"]]) if meta["chave"] < len(linha) else ""
                if ent:
                    b.adicionar(ent, "é_um", classe, 1.0, "observado")
            meta["classe_fixa"] = classe
            acum.append(f"Cada linha agora é um(a) {classe}.")
        self.ia.pensar(verbose=False)
        self.d.mudou = True
        return acum


@recurso
class Formas(Recurso):
    """Campos obrigatórios por classe: 'todo notebook precisa de patrimônio, setor e responsável'."""
    nome = "formas"
    tipos = ("obrigatorio", "incompletos", "campos")

    def interpretar(self, itp, bruto, juntas):
        m = re.match(r"^(?:todo|toda|todos|todas|cada)\s+(.+?)\s+(?:precisa|precisam|deve|devem|"
                     r"tem que|tem de|necessita|necessitam|exige|exigem)\s+"
                     r"(?:de\s+|ter\s+|possuir\s+)?(.+)$", juntas)
        if m:
            classe = itp.nome_entidade(m.group(1).split())
            campos = [self.base.normalizar_relacao(_nome_rel(c))
                      for c in re.split(r"\s*,\s*|\s+e\s+", m.group(2)) if c.strip()]
            if classe and campos:
                return {"tipo": "obrigatorio", "classe": classe, "campos": campos}
        if re.match(r"^(?:o que falta (?:cadastrar|preencher)|o que esta incompleto|"
                    r"quais (?:cadastros|itens) (?:estao|ficaram) incompletos?|cadastros incompletos|"
                    r"o que falta nos cadastros)$", juntas):
            return {"tipo": "incompletos"}
        if juntas in ("campos obrigatorios", "quais os campos obrigatorios", "obrigatorios"):
            return {"tipo": "campos"}
        return None

    def faltantes(self):
        """(entidade, classe, campo) de cada campo obrigatório ainda sem valor."""
        b = self.base
        achados = []
        for classe, campos in sorted(b.formas.items()):
            for f in b.consultar(r="é_um", o=classe, minimo=LIMIAR_ACEITE):
                for campo in campos:
                    if campo not in b.config.get("chaves", [])                             and not b.consultar(s=f.s, r=campo, minimo=LIMIAR_ACEITE):
                        achados.append((f.s, classe, campo))
        return sorted(set(achados))

    def tratar(self, intencao):
        b = self.base
        if intencao["tipo"] == "campos":
            if not b.formas:
                return ["Ainda não há campos obrigatórios. Diga, por exemplo: "
                        "todo notebook precisa de patrimônio, setor e responsável."]
            return ["Campos obrigatórios:"] + [f"  {c}: {', '.join(_legivel(x) for x in rs)}"
                                                for c, rs in sorted(b.formas.items())]
        if intencao["tipo"] == "incompletos":
            falta = self.faltantes()
            if not falta:
                return ["Todos os cadastros têm os campos obrigatórios preenchidos."
                        if b.formas else "Ainda não há campos obrigatórios definidos."]
            self.d._lembrar(falta[0][0])
            linhas = [f"{len({e for e, _, _ in falta})} cadastro(s) incompleto(s):"]
            por_e = defaultdict(list)
            for e, _, campo in falta:
                por_e[e].append(_legivel(campo))
            for e, campos in list(por_e.items())[:10]:
                linhas.append(f"  {_legivel(e)}: falta {', '.join(campos)}")
            if len(por_e) > 10:
                linhas.append(f"  e mais {len(por_e) - 10} cadastro(s).")
            return linhas
        classe, campos = intencao["classe"], intencao["campos"]
        atuais = b.formas.setdefault(classe, [])
        novos = [c for c in campos if c not in atuais]
        atuais.extend(novos)
        self.d.mudou = True
        self.d.registrar("regra", f"{classe} exige {', '.join(campos)}")
        pend = len([x for x in self.faltantes() if x[1] == classe])
        nomes = ", ".join(_legivel(c) for c in atuais)
        return [f"Anotei: todo(a) {_legivel(classe)} precisa de {nomes}."
                + (f" Hoje faltam {pend} campo(s) a preencher; vou perguntar." if pend else "")]

    def perguntas(self, foco):
        falta = self.faltantes()
        saida = []
        for e, classe, campo in falta[:200]:
            saida.append({"score": 1.1, "chave": ("campo", e, campo), "gerador": self._q_campo(e, campo),
                          "entidade": e,
                          "motivo": f"{_legivel(classe)} exige {_legivel(campo)} e {_legivel(e)} ainda não tem"})
            if len(saida) >= 3 and foco is None:
                break
        return saida

    def _q_campo(self, e, campo):
        resp = yield (f"Qual é o(a) {_legivel(campo)} de {_legivel(e)}? "
                      "(Escreva o valor, ou não sei para deixar em aberto.)", "livre")
        if classificar_resposta(resp) in ("nao", "nao_sei") or not resp.strip():
            return ["Tudo bem, deixo em aberto."]
        valor = _valor(resp.strip().rstrip("."))
        self.d._lembrar(e)
        acum = []
        yield from self.d._delegar(self.d._afirmar(e, campo, valor, False, 1.0), acum)
        return acum

    def auditoria(self):
        falta = self.faltantes()
        if not falta:
            return []
        por_e = sorted({e for e, _, _ in falta})
        return [f"  {len(por_e)} cadastro(s) com campos obrigatórios em branco ({len(falta)} campo(s)): "
                + ", ".join(_legivel(e) for e in por_e[:6]) + ("..." if len(por_e) > 6 else ".")]


# Modelos de domínio: só a estrutura (classes, campos obrigatórios, relações com valor único).
# Nenhum valor é preenchido aqui: os dados vêm da pessoa ou dos arquivos dela.
MODELOS = {
    "inventario": {
        "titulo": "Inventário de bens",
        "subclasses": {"Computador": "Bem", "Monitor": "Bem", "Impressora": "Bem",
                       "Mesa": "Bem", "Cadeira": "Bem"},
        "obrigatorios": {"Bem": ["patrimonio", "setor", "responsavel", "numero_de_serie",
                                 "situacao"],
                         "Computador": ["marca", "modelo"]},
        "funcionais": ["patrimonio", "numero_de_serie", "setor", "responsavel", "situacao"],
        "disjuntas": [("Computador", "Mesa"), ("Monitor", "Impressora")],
    },
    "biblioteca": {
        "titulo": "Acervo de uma biblioteca",
        "subclasses": {"Romance": "Livro", "Dicionario": "Livro", "Livro": "Publicacao",
                       "Revista": "Publicacao"},
        "obrigatorios": {"Livro": ["isbn", "titulo", "autor", "editora", "ano"],
                         "Exemplar": ["livro", "tombo", "situacao"]},
        "funcionais": ["isbn", "titulo", "editora", "ano", "tombo", "situacao"],
        "disjuntas": [("Livro", "Revista")],
    },
}


def _nome_modelo(texto):
    return re.sub(r"[\W_]+", " ", _sem_acento(texto)).strip()


@recurso
class Modelos(Recurso):
    nome = "modelos"
    tipos = ("modelos", "aplicar_modelo")

    def interpretar(self, itp, bruto, juntas):
        if juntas in ("modelos", "quais modelos", "quais modelos existem", "lista de modelos",
                      "modelos de dominio", "modelos disponiveis"):
            return {"tipo": "modelos"}
        m = re.match(r"^(?:use|usar|aplique|aplicar|carregue|carregar|crie|criar|ative|ativar)"
                     r"\s+(?:o\s+)?modelo\s+(?:de\s+)?(.+)$", juntas)
        if m:
            alvo = _nome_modelo(m.group(1))
            for chave in MODELOS:
                if _nome_modelo(chave.replace("_", " ")) == alvo:
                    return {"tipo": "aplicar_modelo", "modelo": chave}
            return {"tipo": "modelos", "desconhecido": alvo}
        return None

    def tratar(self, intencao):
        if intencao["tipo"] == "modelos":
            linhas = []
            if intencao.get("desconhecido"):
                linhas.append(f"Não conheço o modelo '{intencao['desconhecido']}'.")
            linhas.append("Modelos disponíveis (escreva usar modelo e o nome):")
            linhas += [f"  {k}: {m['titulo']}" for k, m in MODELOS.items()]
            return linhas
        return self.aplicar(intencao["modelo"])

    def aplicar(self, nome):
        modelo, b = MODELOS[nome], self.base
        anterior = b.proc_atual
        b.proc_atual = {"via": "modelo", "arquivo": f"modelo {nome}"}
        try:
            for sub, sup in modelo["subclasses"].items():
                b.adicionar(sub, "subclasse_de", sup, 1.0, "observado")
        finally:
            b.proc_atual = anterior
        for classe, campos in modelo["obrigatorios"].items():
            lista = b.formas.setdefault(classe, [])
            lista += [c for c in campos if c not in lista]
        for r in modelo["funcionais"]:
            b.declarar_funcional(r)
        for a, c in modelo["disjuntas"]:
            b.declarar_disjuntas(a, c)
        self.ia.pensar(verbose=False)
        self.d.mudou = True
        self.d.registrar("regra", f"Modelo {nome} aplicado")
        linhas = [f"Apliquei o modelo '{nome}' ({modelo['titulo']}).",
                  "Classes: " + ", ".join(sorted(set(modelo["subclasses"]) | set(modelo["subclasses"].values())
                                                  | set(modelo["obrigatorios"]))) + "."]
        for classe, campos in modelo["obrigatorios"].items():
            linhas.append(f"  {classe} exige: {', '.join(_legivel(c) for c in campos)}")
        linhas.append("Só criei a estrutura. Não preenchi valores, números de processo, códigos "
                      "CATMAT/CATSER nem dispositivos legais: isso vem de você ou dos seus arquivos.")
        linhas.append("Carregue uma planilha ou me conte um item, por exemplo: o notebook 1234 é um Notebook.")
        return linhas


# %% [24] Recursos: vocabulário que aprende, modos de raciocínio e propagação
# Vocabulário: a pessoa ensina sinônimos ("ataca significa caça") e o Info os guarda na base.
# Modos: conjunção por produto (probabilidade) ou por mínimo (lógica fuzzy), e acúmulo de
# evidências por máximo ou por ruído-OU. Propagação: mostra em quantas rodadas o raciocínio
# chegou ao ponto fixo e quais conclusões se apoiam em mais de um caminho.

def termo_fuzzy(c):
    """Termo linguístico para uma confiança (conjuntos de pertinência por faixas)."""
    for limite, termo in ((0.90, "quase certo"), (0.70, "provável"), (0.50, "possível"),
                          (0.30, "duvidoso")):
        if c >= limite:
            return termo
    return "improvável"


@recurso
class Vocabulario(Recurso):
    nome = "vocabulario"
    tipos = ("sinonimo",)

    def interpretar(self, itp, bruto, juntas):
        m = (re.match(r"^quando eu (?:digo|falo|escrevo) (.+?) (?:eu )?(?:quero dizer|significa|"
                      r"estou dizendo) (.+)$", juntas)
             or re.match(r"^(.+?) (?:significa|quer dizer|equivale a|e o mesmo que|"
                         r"vale o mesmo que) (.+)$", juntas))
        if not m:
            return None
        frase, alvo = m.group(1).strip(), m.group(2).strip().split()
        if not frase or len(frase.split()) > 4 or not alvo:
            return None
        achado = itp._achar_relacao(alvo, itp.lexico(), False)
        if achado and achado[0] == 0 and achado[1] == len(alvo):
            return {"tipo": "sinonimo", "frase": frase, "relacao": achado[2]}
        return None

    def tratar(self, intencao):
        frase, rel = intencao["frase"], intencao["relacao"]
        self.base.vocabulario[frase] = rel
        self.d.mudou = True
        self.d.registrar("vocabulario", f"{frase} = {rel}")
        return [f"Aprendi: quando você escrever '{frase}', entendo '{SUPERFICIE.get(rel, _legivel(rel))}' "
                f"({rel}). Isso fica gravado na base."]

    def apos_nova_relacao(self, s, nome, o):
        resp = yield (f"A relação '{_legivel(nome)}' tem o mesmo sentido de alguma que eu já conheço? "
                      "Se tiver, diga qual (por exemplo: caça). Se não tiver, diga não.", "livre")
        if classificar_resposta(resp) in ("nao", "nao_sei"):
            return ["Certo, fica como uma relação própria."]
        itp = self.d.interprete
        norm = [_sem_acento(t) for t in itp.tokens(resp)]
        lex = {k: v for k, v in itp.lexico().items() if v != nome}
        achado = itp._achar_relacao(norm, lex, False)
        if not achado:
            return [f"Não reconheci essa relação. Mantenho '{_legivel(nome)}' como relação própria."]
        rel = achado[2]
        b = self.base
        b.vocabulario[_sem_acento(_legivel(nome))] = rel
        for f in [f for f in b.fatos.values() if f.r == nome and f.fonte in FONTES_FIRMES]:
            b.remover(f.chave)
            novo = b.adicionar(f.s, rel, f.o, 0.5, f.fonte)
            if novo is not None:
                novo.alpha, novo.beta = f.alpha, f.beta
                novo.procedencias, novo.valido_ate = f.procedencias, f.valido_ate
        self.ia.pensar(verbose=False)
        self.d.mudou = True
        self.d.registrar("vocabulario", f"{nome} = {rel}")
        return [f"Entendi: '{_legivel(nome)}' passa a valer como '{SUPERFICIE.get(rel, _legivel(rel))}'."]


@recurso
class Modos(Recurso):
    nome = "modos"
    tipos = ("modo",)

    def interpretar(self, itp, bruto, juntas):
        mapa = (
            (r"^(?:use|usar|ative|ativar|mude para|modo)\s+(?:a\s+)?(?:logica\s+)?(?:fuzzy|nebulosa)$",
             {"conjuncao": "minimo", "rotulos_fuzzy": True}),
            (r"^(?:use|usar|ative|ativar|volte para|mude para|modo)\s+(?:a\s+)?(?:logica\s+)?"
             r"(?:probabilidade|probabilistic[ao]|probabilistica)$",
             {"conjuncao": "produto", "rotulos_fuzzy": False}),
            (r"^(?:acumule|acumular|some|somar)\s+(?:as\s+)?evidencias$",
             {"acumulo": "ruido_ou"}),
            (r"^(?:use|usar)\s+o\s+melhor\s+caminho$|^nao\s+acumule\s+evidencias$",
             {"acumulo": "maximo"}))
        for padrao, ajuste in mapa:
            if re.match(padrao, juntas):
                return {"tipo": "modo", "ajuste": ajuste}
        if juntas in ("como voce raciocina", "qual o modo", "modos de raciocinio", "qual modo voce usa"):
            return {"tipo": "modo", "ajuste": {}}
        return None

    def tratar(self, intencao):
        b = self.base
        ajuste = intencao["ajuste"]
        antes = {f.chave: f.confianca for f in b.fatos.values() if f.fonte == "inferido"}
        b.config.update(ajuste)
        if ajuste:
            self.ia.pensar(verbose=False)
            self.d.mudou = True
        depois = {f.chave: f.confianca for f in b.fatos.values() if f.fonte == "inferido"}
        linhas = []
        if ajuste.get("conjuncao") == "minimo":
            linhas.append("Passei para a lógica fuzzy: a confiança de uma conclusão passa a ser a da "
                          "premissa mais fraca (o mínimo), em vez do produto delas. Vou descrever "
                          "confianças também em palavras (quase certo, provável, possível...).")
        elif ajuste.get("conjuncao") == "produto":
            linhas.append("Voltei para a probabilidade: as confianças das premissas se multiplicam.")
        if ajuste.get("acumulo") == "ruido_ou":
            linhas.append("Agora evidências independentes se somam: se duas regras chegam à mesma "
                          "conclusão, a confiança é 1 menos o produto das dúvidas.")
        elif ajuste.get("acumulo") == "maximo":
            linhas.append("Voltei a valer só o melhor caminho para cada conclusão.")
        if ajuste:
            mudaram = [k for k in depois if abs(depois[k] - antes.get(k, 0)) >= 0.05]
            novos = [k for k in depois if k not in antes]
            linhas.append(f"Isso alterou a confiança de {len(mudaram) - len(novos)} dedução(ões) e fez "
                          f"surgir {len(novos)}." if (mudaram or novos) else
                          "Nenhuma dedução mudou com essa troca nesta base.")
        else:
            c, a = b.config.get("conjuncao", "produto"), b.config.get("acumulo", "maximo")
            linhas.append("Modo atual: conjunção por " + ("mínimo (fuzzy)" if c == "minimo" else "produto "
                          "(probabilidade)") + "; acúmulo por " + ("ruído-OU (evidências se somam)"
                          if a == "ruido_ou" else "máximo (vale o melhor caminho)") + ".")
        return linhas


@recurso
class Propagacao(Recurso):
    nome = "propagacao"
    tipos = ("propagacao",)

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:mostre|mostra|explique|como e|veja)?\s*(?:a\s+)?propagacao(?: das conclusoes)?$|"
                    r"^como as conclusoes se propagam$|^mostre as rodadas$", juntas):
            return {"tipo": "propagacao"}
        return None

    def tratar(self, intencao):
        motor = self.ia.motor
        self.ia.pensar(verbose=False)
        ger = motor.geracoes
        if not ger:
            return ["Não deduzi nada ainda, então não há propagação para mostrar."]
        linhas = [f"O raciocínio chegou ao ponto fixo em {len(ger)} rodada(s):"]
        linhas += [f"  rodada {i}: {n} conclusão(ões) nova(s)" for i, n in enumerate(ger, 1)]
        multi = sorted(((n, k) for k, n in motor.n_derivacoes.items() if n >= 2), reverse=True)
        b = self.base
        if multi:
            linhas.append("Conclusões que se apoiam em mais de um caminho:")
            for n, k in multi[:5]:
                f = b.fatos.get(k)
                if f is not None:
                    linhas.append(f"  {frase_fato(*k)}: {n} caminhos, confiança {f.confianca:.0%}")
            if b.config.get("acumulo") != "ruido_ou":
                linhas.append("Hoje vale só o melhor caminho. Escreva acumular evidências para somá-los.")
        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(5.6, 2.6), dpi=130, facecolor="white")
            ax = fig.subplots()
            ax.bar(range(1, len(ger) + 1), ger, color=MARCA["acento_claro"], width=0.55)
            ax.set_xticks(range(1, len(ger) + 1))
            ax.set_xlabel("rodada de dedução", fontsize=8)
            ax.set_ylabel("conclusões novas", fontsize=8)
            ax.set_title(f"{MARCA['empresa']} | {MARCA['ia']}: propagação", fontsize=9,
                         color=MARCA["acento_claro"], loc="left")
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            fig.savefig("propagacao.png", bbox_inches="tight")
            linhas.insert(1, "[[img:propagacao.png]]")
        except Exception:
            pass
        return linhas


# %% [25] Recursos: dados com prazo de validade
# Um fato pode valer só até uma data ("o contrato vale até 31/12/2026"). O Info avisa do que
# venceu ou está perto de vencer e pergunta se ainda vale. Fatos vencidos continuam na base
# até a pessoa decidir; a pergunta é o mecanismo, não uma exclusão automática.

AVISO_DIAS = 30


def _data(iso):
    try:
        return date.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


@recurso
class Tempo(Recurso):
    nome = "tempo"
    tipos = ("vencimentos",)

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:o que|quais?) (?:esta |estao |ja )?(?:vence|vencem|vencido|vencidos|vai vencer|"
                    r"vao vencer|venceu|venceram)(?: .*)?$|^vencimentos$|^prazos$|^o que vence$", juntas):
            return {"tipo": "vencimentos"}
        frase, ate = separar_validade(bruto)
        if ate:
            sub = itp.interpretar(frase)
            if sub["tipo"] == "afirmar":
                sub["ate"] = ate
                return sub
        return None

    def _situacao(self):
        hoje = self.d._hoje()
        vencidos, proximos = [], []
        for f in self.base.fatos.values():
            alvo = _data(f.valido_ate) if f.fonte in FONTES_FIRMES else None
            if alvo is None:
                continue
            dias = (alvo - hoje).days
            if dias < 0:
                vencidos.append((dias, f))
            elif dias <= AVISO_DIAS:
                proximos.append((dias, f))
        return sorted(vencidos, key=lambda x: x[0]), sorted(proximos, key=lambda x: x[0])

    def tratar(self, intencao):
        vencidos, proximos = self._situacao()
        if not vencidos and not proximos:
            comprazo = sum(1 for f in self.base.fatos.values() if f.valido_ate)
            return [f"Nada vencido nem vencendo nos próximos {AVISO_DIAS} dias."
                    + (f" ({comprazo} fato(s) têm prazo.)" if comprazo else "")]
        linhas = []
        if vencidos:
            linhas.append(f"Vencidos ({len(vencidos)}):")
            linhas += [f"  {frase_fato(*f.chave)} (venceu em {_data_br(f.valido_ate)})"
                       for _, f in vencidos[:10]]
        if proximos:
            linhas.append(f"Vencem nos próximos {AVISO_DIAS} dias ({len(proximos)}):")
            linhas += [f"  {frase_fato(*f.chave)} (em {d} dia(s), {_data_br(f.valido_ate)})"
                       for d, f in proximos[:10]]
        return linhas

    def perguntas(self, foco):
        vencidos, proximos = self._situacao()
        saida = []
        for pontos, grupo in ((1.0, vencidos), (0.7, proximos)):
            for dias, f in grupo[:5]:
                saida.append({"score": pontos, "chave": ("validade", f.chave), "entidade": f.s,
                              "gerador": self._q_validade(f, dias),
                              "motivo": "o prazo de validade desse fato venceu" if dias < 0
                              else "o prazo de validade desse fato está acabando"})
        return saida

    def _q_validade(self, f, dias):
        chave = f.chave
        if dias < 0:
            texto = (f"{frase_fato(*chave)} valia até {_data_br(f.valido_ate)} e esse prazo já "
                     "passou. Isso ainda vale?")
        else:
            texto = (f"{frase_fato(*chave)} vale até {_data_br(f.valido_ate)} (daqui a {dias} dia(s)). "
                     "Já foi renovado ou prorrogado?")
        resp = yield (texto, "sim_nao")
        if resp == "nao":
            if dias < 0:
                self.base.remover(chave)
                self.ia.pensar(verbose=False)
                self.d.mudou = True
                self.d.registrar("decisao", f"Fato vencido removido: {frase_fato(*chave)}")
                return ["Entendi. Tirei esse fato da base."]
            return ["Certo, mantenho o prazo atual."]
        if resp != "sim":
            return ["Tudo bem, deixo para depois."]
        nova = yield ("Até quando vale agora? Escreva a data (DD/MM/AAAA) ou sem prazo.", "livre")
        m = re.search(r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}", nova)
        if m and _data_iso(m.group(0)):
            f.valido_ate = _data_iso(m.group(0))
            resultado = f"Anotei: vale até {_data_br(f.valido_ate)}."
        elif re.search(r"sem\s+prazo|indeterminad|nao\s+tem|não\s+tem", nova, re.I):
            f.valido_ate = None
            resultado = "Anotei: sem prazo de validade."
        else:
            return ["Não entendi a data; mantive o prazo anterior."]
        self.d.mudou = True
        self.d.registrar("decisao", f"Prazo atualizado: {frase_fato(*chave)}")
        return [resultado]

    def auditoria(self):
        vencidos, proximos = self._situacao()
        linhas = []
        if vencidos:
            linhas.append(f"  {len(vencidos)} fato(s) com prazo vencido, por exemplo: "
                          + frase_fato(*vencidos[0][1].chave) + ".")
        if proximos:
            linhas.append(f"  {len(proximos)} fato(s) vencem nos próximos {AVISO_DIAS} dias.")
        return linhas


# %% [26] Recursos: calibração por algoritmo genético
# Cada vez que a pessoa responde sim ou não a uma dedução, nasce um rótulo (fato, verdade).
# Com rótulos suficientes, um algoritmo genético procura os parâmetros do raciocínio que
# melhor prevêem o julgamento dela: o fator de confiança das regras manuais/induzidas e o da
# herança, a conjunção (produto ou mínimo) e o acúmulo (máximo ou ruído-OU). A nota é o erro
# de Brier (média de (confiança - verdade)^2): quanto menor, melhor. Cada candidato é avaliado
# numa cópia da base em que o efeito do próprio rótulo foi desfeito, senão a resposta da
# pessoa "vazaria" para a previsão. Fica de fora o ajuste que o feedback fez na confiança das
# regras, o que torna a nota um pouco otimista.

import copy

MIN_ROTULOS = 8
LIMITE_TEMPO_GA = 20.0


def _clone_sem_rotulos(ia, rotulos):
    clone = copy.deepcopy(ia)
    b = clone.base
    for s_, r_, o_, verdade in rotulos:
        chave = (s_, r_, o_)
        b.rejeitados.discard(chave)
        f = b.fatos.get(chave)
        if verdade and f is not None and f.fonte == "confirmado":
            b.remover(chave)
        if not verdade:
            oposta = b.relacao_oposta(r_)
            contra = b.fatos.get((s_, oposta, o_)) if oposta else None
            if (contra is not None and contra.fonte == "observado"
                    and all(not p.get("arquivo") for p in contra.procedencias)):
                b.remover(contra.chave)
    return clone


def _brier(clone, genoma, rotulos):
    b = clone.base
    b.config["fatores"] = {"usuario": genoma[0], "heranca": genoma[1]}
    b.config["conjuncao"] = "minimo" if genoma[2] >= 0.5 else "produto"
    b.config["acumulo"] = "ruido_ou" if genoma[3] >= 0.5 else "maximo"
    b.bloqueados = set()
    clone.motor.executar()
    erro = 0.0
    for s_, r_, o_, verdade in rotulos:
        f = b.fatos.get((s_, r_, o_))
        p = f.confianca if f is not None and f.fonte != "observado" else 0.0
        erro += (p - verdade) ** 2
    return erro / len(rotulos)


def calibrar(ia, rotulos, populacao=12, geracoes=10, tempo=LIMITE_TEMPO_GA, semente=7):
    """Devolve (genoma, brier_antes, brier_depois, geracoes_rodadas, janela_de_tempo_estourou)."""
    rng = random.Random(semente)
    clone = _clone_sem_rotulos(ia, rotulos)
    inicio = time.time()
    neutro = [1.0, 1.0, 0.0, 0.0]
    nota = {}

    def pontuar(g):
        chave = tuple(round(x, 3) for x in g)
        if chave not in nota:
            nota[chave] = _brier(clone, g, rotulos) + 0.01 * (abs(g[0] - 1) + abs(g[1] - 1))
        return nota[chave]
    antes = _brier(clone, neutro, rotulos)
    pop = [neutro] + [[rng.uniform(0.6, 1.3), rng.uniform(0.6, 1.3), rng.random(), rng.random()]
                      for _ in range(populacao - 1)]
    rodadas, estourou = 0, False
    for _ in range(geracoes):
        if time.time() - inicio > tempo:
            estourou = True
            break
        pop.sort(key=pontuar)
        elite = pop[:max(2, populacao // 4)]
        filhos = [list(g) for g in elite]
        while len(filhos) < populacao:
            pai = min(rng.sample(pop, 3), key=pontuar)
            mae = min(rng.sample(pop, 3), key=pontuar)
            filho = [pai[i] if rng.random() < 0.5 else mae[i] for i in range(4)]
            for i in (0, 1):
                if rng.random() < 0.4:
                    filho[i] = min(1.4, max(0.4, filho[i] + rng.gauss(0, 0.12)))
            for i in (2, 3):
                if rng.random() < 0.15:
                    filho[i] = 1.0 - filho[i] if filho[i] in (0.0, 1.0) else rng.random()
            filhos.append(filho)
        pop = filhos
        rodadas += 1
    melhor = min(pop, key=pontuar)
    depois = _brier(clone, melhor, rotulos)
    return melhor, antes, depois, rodadas, estourou


@recurso
class Genetico(Recurso):
    nome = "genetico"
    tipos = ("calibrar",)

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:calibre|calibrar|otimize|otimizar|ajuste|ajustar|evolua|evoluir|"
                    r"treine|treinar)\s+(?:os\s+|as\s+|a\s+|o\s+)?(?:parametros|confiancas?|"
                    r"modelo|raciocinio|algoritmo genetico)$|^use o algoritmo genetico$", juntas):
            return {"tipo": "calibrar"}
        return None

    def tratar(self, intencao):
        return self._calibrar()

    def _calibrar(self):
        rotulos = [x for x in self.base.rotulos if len(x) == 4]
        if len(rotulos) < MIN_ROTULOS:
            return [f"Tenho só {len(rotulos)} julgamento(s) seu(s) sobre as minhas deduções e preciso de "
                    f"pelo menos {MIN_ROTULOS}. Eles surgem quando você responde sim ou não às minhas "
                    "perguntas de verificação."]
        ok = yield (f"Vou procurar os parâmetros que melhor prevêem os seus {len(rotulos)} julgamentos "
                    "(até 20 segundos). Pode seguir?", "sim_nao")
        if ok != "sim":
            return ["Certo, deixo como está."]
        melhor, antes, depois, rodadas, estourou = calibrar(self.ia, rotulos)
        if depois >= antes - 1e-4:
            return [f"Testei variações por {rodadas} geração(ões) e não achei nada melhor que o atual "
                    f"(erro de Brier {antes:.3f}). Mantenho como está."]
        resumo = (f"regras suas x{melhor[0]:.2f}, herança x{melhor[1]:.2f}, conjunção por "
                  f"{'mínimo' if melhor[2] >= 0.5 else 'produto'}, acúmulo por "
                  f"{'soma de evidências' if melhor[3] >= 0.5 else 'melhor caminho'}")
        resp = yield (f"Em {rodadas} geração(ões)" + (" (parei no limite de tempo)" if estourou else "")
                      + f", o erro de previsão caiu de {antes:.3f} para {depois:.3f} (Brier, menor é "
                      f"melhor). Parâmetros: {resumo}. Aplico?", "sim_nao")
        if resp != "sim":
            return ["Certo, não apliquei nada."]
        b = self.base
        b.config["fatores"] = {"usuario": round(melhor[0], 3), "heranca": round(melhor[1], 3)}
        b.config["conjuncao"] = "minimo" if melhor[2] >= 0.5 else "produto"
        b.config["acumulo"] = "ruido_ou" if melhor[3] >= 0.5 else "maximo"
        self.ia.pensar(verbose=False)
        self.d.mudou = True
        self.d.registrar("calibracao", resumo, brier_antes=antes, brier_depois=depois)
        return ["Apliquei os parâmetros calibrados e refiz as deduções."]


# %% [27] Recursos: métricas, avaliação das respostas e motivo das perguntas
# O diário da sessão (Dialogo.diario) guarda o que aconteceu. Daí saem as métricas: quantas
# deduções e sugestões a pessoa aceitou ou recusou, e quão bem a confiança atual prevê os
# julgamentos dela. "boa resposta" e "resposta ruim" registram a opinião sobre a última fala
# da IA, sem botões: o Info não tem menus.

@recurso
class Qualidade(Recurso):
    nome = "qualidade"
    tipos = ("metricas", "avaliar", "motivo")

    def interpretar(self, itp, bruto, juntas):
        if juntas in ("metricas", "estatisticas", "como voce esta indo", "desempenho",
                      "mostre as metricas", "quais as metricas"):
            return {"tipo": "metricas"}
        if juntas in ("boa resposta", "resposta boa", "otima resposta", "gostei", "gostei da resposta"):
            return {"tipo": "avaliar", "valor": 1}
        if juntas in ("resposta ruim", "ma resposta", "resposta errada", "resposta fraca"):
            return {"tipo": "avaliar", "valor": -1}
        if re.match(r"^por ?que (?:voce )?(?:me )?(?:perguntou|pergunta|esta perguntando|perguntando) "
                    r"(?:isso|isto|essa pergunta|esse assunto)$|^por ?que essa pergunta$", juntas):
            return {"tipo": "motivo"}
        return None

    def tratar(self, intencao):
        tipo = intencao["tipo"]
        if tipo == "motivo":
            return self.d._explicar_motivo()
        if tipo == "avaliar":
            valor = intencao["valor"]
            resposta = " ".join(self.d.ultima)[:200]
            self.d.registrar("avaliacao", resposta, valor=valor)
            return ["Obrigado, anotei que a resposta foi boa." if valor > 0 else
                    "Obrigado, anotei que a resposta foi ruim. Se quiser, diga o que estava errado "
                    "e eu corrijo na base."]
        return self.metricas()

    def metricas(self):
        d, b = self.d, self.base
        ev = lambda t: [e for e in d.diario if e["tipo"] == t]
        linhas = ["Métricas desta sessão:"]
        linhas.append(f"  {len(ev('fala'))} fala(s) suas, {len(ev('anotado'))} fato(s) anotado(s), "
                      f"{len(ev('pergunta'))} pergunta(s) minha(s) por iniciativa.")
        for tipo, rotulo in (("verificacao", "deduções que verifiquei"),
                             ("sugestao", "palpites que sugeri")):
            lista = ev(tipo)
            if lista:
                ok = sum(1 for e in lista if e.get("aceita"))
                linhas.append(f"  {rotulo}: {ok} aceita(s) e {len(lista) - ok} recusada(s) "
                              f"({ok / len(lista):.0%} de acerto).")
        avaliacoes = ev("avaliacao")
        if avaliacoes:
            boas = sum(1 for e in avaliacoes if e["valor"] > 0)
            linhas.append(f"  avaliações das respostas: {boas} boa(s) e {len(avaliacoes) - boas} ruim(ns).")
        rotulos = [x for x in b.rotulos if len(x) == 4]
        if rotulos:
            # avaliado numa cópia sem o efeito das próprias respostas, senão o erro seria ~0
            cfg = b.config
            genoma = [cfg.get("fatores", {}).get("usuario", 1.0), cfg.get("fatores", {}).get("heranca", 1.0),
                      1.0 if cfg.get("conjuncao") == "minimo" else 0.0,
                      1.0 if cfg.get("acumulo") == "ruido_ou" else 0.0]
            erro = _brier(_clone_sem_rotulos(self.ia, rotulos), genoma, rotulos)
            linhas.append(f"  erro de previsão (Brier) sobre {len(rotulos)} julgamento(s) seu(s): "
                          f"{erro:.3f} (0 é perfeito; 0,25 equivale a chutar).")
        if b.fontes_stats:
            fracas = sorted(b.fontes_stats, key=b.peso_fonte)[:2]
            linhas.append("  fontes com menor peso: "
                          + ", ".join(f"{k} ({b.peso_fonte(k):.2f})" for k in fracas) + ".")
        if len(linhas) == 2 and not ev("verificacao") and not ev("sugestao"):
            linhas.append("  Ainda não há respostas suas a verificações para medir.")
        return linhas


# %% [28] Recursos: retomar a sessão e ata da conversa
# Sessão: cada gravação deixa um info_sessao.json ao lado da base com o que estava aberto. Na
# próxima abertura sem arquivo, o Info oferece retomar de onde parou. Ata: um PDF com o que
# foi carregado, anotado, decidido e o que ficou pendente, montado a partir do diário.

ARQUIVO_SESSAO = "info_sessao.json"


@recurso
class Sessao(Recurso):
    nome = "sessao"

    def ao_salvar(self, destino):
        firmes = sum(1 for f in self.base.fatos.values() if f.fonte in FONTES_FIRMES)
        dados = {"destino": os.path.abspath(destino), "arquivo": self.d.arquivo,
                 "quando": datetime.now().isoformat(timespec="seconds"), "fatos": firmes,
                 "pendente": self.d.ultimo_motivo}
        try:
            pasta = os.path.dirname(os.path.abspath(destino))
            with open(os.path.join(pasta, ARQUIVO_SESSAO), "w", encoding="utf-8") as fp:
                json.dump(dados, fp, ensure_ascii=False, indent=1)
        except OSError:
            pass  # sem permissão de escrita: perde só a retomada, não a base

    def _ler(self):
        for pasta in (os.getcwd(),):
            caminho = os.path.join(pasta, ARQUIVO_SESSAO)
            if os.path.exists(caminho):
                try:
                    with open(caminho, encoding="utf-8") as fp:
                        dados = json.load(fp)
                except (OSError, ValueError):
                    continue
                if os.path.exists(dados.get("destino", "")):
                    return dados
        return None

    def ao_abrir(self):
        if self.base.fatos:
            return []
        dados = self._ler()
        if dados:
            self.d.fila.append(self._q_retomar(dados))
        return []

    def _q_retomar(self, dados):
        nome = os.path.basename(dados["destino"])
        resp = yield (f"Na última vez você estava trabalhando em '{nome}' ({dados.get('fatos', '?')} "
                      f"fatos, {_momento(dados.get('quando'))}). Quer continuar dele?", "sim_nao")
        if resp != "sim":
            return ["Certo, começamos do zero."]
        linhas = self.d.carregar(dados["destino"])
        return ["Retomando de onde parou."] + linhas


class _Docx:
    """Escritor mínimo de Word (python-docx), com a mesma interface do _Pdf."""

    def __init__(self):
        import docx
        self.doc = docx.Document()
        self.cor = None

    def linha(self, texto, fonte="helv", tam=10.5, cor=None, antes=0):
        if tam >= 16:
            self.doc.add_heading(texto, 0)
        elif fonte == "hebo" and tam >= 13:
            self.doc.add_heading(texto, 1)
        else:
            self.doc.add_paragraph(texto)

    def titulo(self, texto):
        self.doc.add_heading(texto, 1)

    def paragrafo(self, texto, prefixo="", **kw):
        self.doc.add_paragraph(prefixo + texto)

    def salvar(self, caminho):
        self.doc.save(caminho)


class _Pdf:
    """Escritor mínimo de PDF com a identidade da marca (usa PyMuPDF)."""

    def __init__(self):
        self.fitz = _importar_fitz()
        self.doc = self.fitz.open()
        self.pg, self.y = None, 0
        self.cor = _rgb(MARCA["acento_claro"])

    def _nova(self):
        self.pg = self.doc.new_page()
        self.pg.insert_text((56, 42), MARCA["empresa"], fontname="hebo", fontsize=11, color=self.cor)
        self.pg.insert_text((118, 42), "| " + MARCA["ia"], fontname="helv", fontsize=11,
                            color=(0.45, 0.45, 0.45))
        self.pg.draw_line((56, 52), (539, 52), color=(0.88, 0.88, 0.88), width=0.6)
        self.y = 82

    def linha(self, texto, fonte="helv", tam=10.5, cor=(0.12, 0.12, 0.12), antes=0):
        if self.pg is None or self.y + antes > 780:
            self._nova()
        self.y += antes
        self.pg.insert_text((56, self.y), texto, fontname=fonte, fontsize=tam, color=cor)
        self.y += tam * 1.5

    def titulo(self, texto):
        self.linha(texto, "hebo", 13, antes=14)

    def paragrafo(self, texto, prefixo="", **kw):
        for i, parte in enumerate(textwrap.wrap(texto, 92 - len(prefixo)) or [""]):
            self.linha((prefixo if i == 0 else " " * len(prefixo)) + parte, **kw)

    def salvar(self, caminho):
        if self.pg is None:
            self._nova()
        self.doc.save(caminho)
        self.doc.close()


@recurso
class Ata(Recurso):
    nome = "ata"
    tipos = ("ata",)

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:gere|gerar|faca|fazer|escreva|crie|criar|monte)?\s*(?:a\s+|uma\s+)?ata"
                    r"(?: da (?:conversa|sessao|reuniao))?$|^relatorio da sessao$", juntas):
            return {"tipo": "ata"}
        return None

    def tratar(self, intencao):
        d, b = self.d, self.base
        ev = lambda *tipos: [e for e in d.diario if e["tipo"] in tipos]
        formato = intencao.get("formato", "pdf")
        if formato == "docx":
            try:
                pdf = _Docx()
            except ImportError:
                return ["Para a ata em Word preciso do pacote python-docx (pip install python-docx)."]
        else:
            pdf = _Pdf()
        pdf.linha("Ata da conversa", "hebo", 18, pdf.cor)
        pdf.linha(f"Gerada em {datetime.now():%d/%m/%Y %H:%M} por {MARCA['ia']}, da {MARCA['empresa']}",
                  cor=(0.4, 0.4, 0.4))
        pdf.titulo("Resumo")
        firmes = sum(1 for f in b.fatos.values() if f.fonte in FONTES_FIRMES)
        pdf.linha(f"Base: {os.path.basename(d.arquivo) if d.arquivo else 'sem arquivo carregado'}; "
                  f"{firmes} fato(s) informado(s), {len(b.fatos) - firmes} deduzido(s).")
        pdf.linha(f"Falas na sessão: {len(ev('fala'))}. Fatos anotados: {len(ev('anotado'))}.")
        cargas = ev("carga")
        if cargas:
            pdf.titulo("Arquivos lidos")
            for e in cargas:
                pdf.paragrafo(f"{_momento(e['quando'])}: {e['texto']}", "- ")
        m = d._mudancas()
        if m["novos"] or m["removidos"] or m["alterados"]:
            pdf.titulo("O que mudou na base")
            for k in m["novos"][:50]:
                pdf.paragrafo(frase_fato(*k), "+ ")
            for k in m["removidos"][:50]:
                pdf.paragrafo(frase_fato(*k), "- ")
            for k, antes, depois in m["alterados"][:50]:
                pdf.paragrafo(f"{frase_fato(*k)} ({antes:.0%} para {depois:.0%})", "~ ")
        decisoes = ev("decisao", "verificacao", "sugestao", "regra", "vocabulario", "calibracao")
        if decisoes:
            pdf.titulo("Decisões e confirmações")
            for e in decisoes[:80]:
                marca = ""
                if "aceita" in e:
                    marca = " (confirmado)" if e["aceita"] else " (recusado)"
                pdf.paragrafo(f"{_momento(e['quando'])}: {e['texto']}{marca}", "- ")
        pend = []
        for r in d.recursos:
            pend += r.auditoria()
        conflitos = self.ia.motor.verificar_consistencia(resolver=False)
        if conflitos:
            pend.append(f"  {len(conflitos)} contradição(ões) ainda sem decisão.")
        pdf.titulo("Pendências")
        if pend:
            for linha in pend:
                pdf.paragrafo(linha.strip(), "- ")
        else:
            pdf.linha("Nenhuma pendência encontrada.")
        perguntas = ev("pergunta")
        if perguntas:
            pdf.titulo("Perguntas que fiz por iniciativa")
            for e in perguntas[:40]:
                pdf.paragrafo(f"Perguntei porque {e['texto']}.", "- ")
        pdf.linha("Gerada automaticamente a partir do diário da sessão; confira antes de circular.",
                  tam=8.5, cor=(0.5, 0.5, 0.5), antes=14)
        caminho = f"{d._stem()}_ata.{formato}"
        try:
            pdf.salvar(caminho)
        except Exception as erro:
            return [f"Não consegui gerar a ata: {erro}"]
        return [f"Gerei a ata da conversa em '{caminho}'.", f"[[baixar:{caminho}]]"]


# %% [29] Recursos: interpretação por modelo de linguagem (opcional, com consentimento)
# O motor simbólico continua sendo a fonte da verdade. O modelo só traduz uma frase que o
# interpretador de regras não entendeu em fatos candidatos; a pessoa confirma cada um antes
# de entrar na base. Fica desligado até a pessoa pedir e autorizar o envio das frases. Usa o
# SDK oficial (pacote anthropic) e a chave em ANTHROPIC_API_KEY. INFO_LLM_URL troca o endereço
# (para testes com servidor local) e INFO_LLM_MODEL troca o modelo.

MODELO_LLM = "claude-opus-5-5"
_PROMPT_LLM = (
    "Você converte uma frase em português em fatos para uma base de conhecimento. "
    "Responda somente com JSON no formato {\"fatos\": [[sujeito, relacao, objeto, negado]]}, "
    "com negado true ou false. Use os nomes de relação já existentes quando servirem: {relacoes}. "
    "Entidades em CamelCase sem acento (por exemplo Gato, DellLatitude). Se a frase não afirmar "
    "nenhum fato, responda {\"fatos\": []}. Não invente nada que a frase não diga."
)


def _cliente_llm():
    try:
        import anthropic
    except ImportError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic"], check=True,
                       timeout=120)
        import anthropic
    url = os.environ.get("INFO_LLM_URL")
    return anthropic.Anthropic(base_url=url) if url else anthropic.Anthropic()


def traduzir_com_llm(frase, relacoes):
    """Pede ao modelo os fatos de uma frase. Devolve lista de (s, r, o, negado)."""
    cliente = _cliente_llm()
    resposta = cliente.messages.create(
        model=os.environ.get("INFO_LLM_MODEL", MODELO_LLM), max_tokens=2000,
        output_config={"effort": "low"},
        system=_PROMPT_LLM.replace("{relacoes}", ", ".join(sorted(relacoes)[:40]) or "nenhuma"),
        messages=[{"role": "user", "content": frase}])
    texto = "".join(b.text for b in resposta.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        return []
    fatos = []
    for item in json.loads(m.group(0)).get("fatos", []):
        if isinstance(item, list) and len(item) >= 3 and all(str(x).strip() for x in item[:3]):
            fatos.append((str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip(),
                          bool(item[3]) if len(item) > 3 else False))
    return fatos


@recurso
class ModeloDeLinguagem(Recurso):
    nome = "llm"
    tipos = ("llm",)

    def __init__(self, dialogo):
        super().__init__(dialogo)
        self.ligado = False

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:ative|ativar|ligue|ligar|use|usar)\s+(?:o\s+)?(?:modo\s+avancado|"
                    r"modelo de linguagem|claude)$", juntas):
            return {"tipo": "llm", "acao": "ligar"}
        if re.match(r"^(?:desative|desativar|desligue|desligar|pare de usar)\s+(?:o\s+)?"
                    r"(?:modo\s+avancado|modelo de linguagem|claude)$", juntas):
            return {"tipo": "llm", "acao": "desligar"}
        return None

    def tratar(self, intencao):
        if intencao["acao"] == "desligar":
            self.ligado = False
            return ["Desativei o modo avançado. Nenhuma frase será enviada."]
        return self._ligar()

    def _ligar(self):
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("INFO_LLM_URL"):
            return ["Para o modo avançado preciso da chave da API na variável de ambiente "
                    "ANTHROPIC_API_KEY. No Colab, guarde-a em Secrets e exporte-a antes de usar. "
                    "Sem ela, sigo só com o interpretador de regras."]
        resp = yield ("O modo avançado envia a um serviço externo (API da Anthropic) as frases que o "
                      "meu interpretador não entender. Os fatos que vierem de lá só entram na base "
                      "depois que você confirmar. Autoriza o envio dessas frases?", "sim_nao")
        if resp != "sim":
            return ["Certo, fica desligado."]
        self.ligado = True
        self.d.registrar("regra", "Modo avançado autorizado")
        return ["Modo avançado ligado. Escreva desativar modo avançado para parar o envio."]

    def sem_entendimento(self, texto):
        if not self.ligado:
            return None
        return self._traduzir(texto)

    def _traduzir(self, texto):
        try:
            fatos = traduzir_com_llm(texto, {f.r for f in self.base.fatos.values()})
        except Exception as erro:
            return ["O modo avançado falhou (" + type(erro).__name__ + "). "
                    "Tente reescrever a frase no formato sujeito, verbo e objeto."]
        itp = self.d.interprete
        fatos = [itp._efetivo(itp.nome_entidade(s.split("_")) or s, r, itp.nome_entidade(o.split("_")) or o, n)
                 for s, r, o, n in fatos]
        if not fatos:
            return ["Nem o modo avançado achou um fato nessa frase."]
        lista = "; ".join(frase_fato(s, r, o) if not n else "não " + frase_fato(s, r, o)
                          for s, r, o, n in fatos)
        resp = yield (f"Entendi como: {lista}. Está certo?", "sim_nao")
        if resp != "sim":
            return ["Certo, não anotei nada."]
        self.d._lembrar(*[f[0] for f in fatos])
        acum = []
        yield from self.d._delegar(self.d._afirmar_varios(fatos, 0.9), acum)
        return acum


# %% [30] Recursos: conversa natural (contexto, tolerância a erros, comparação, hipóteses)
# O que faz o Info parecer um interlocutor e não um formulário:
#   contexto       "e ele vive onde?" e "e o tigre?" reaproveitam o assunto anterior
#   tolerância     "o lião" vira Leão quando a grafia é quase igual a uma entidade conhecida
#   comparação     "o que o gato e o cão têm em comum", "quem se parece com o leão"
#   hipótese       "e se o gato voa?": raciocina numa cópia e mostra o que mudaria
#   narrativa      "fale sobre o leão" escreve parágrafos, com tom conforme a certeza
#   metacognição   "o que você não sabe?": aponta lacunas e deduções frágeis
#   social         apresentação, ajuda, cumprimentos
# Nada aqui altera a base sem a pessoa pedir: a hipótese roda numa cópia.

PALAVRAS_COMUNS = set("""que quem onde como qual quais quando por porque para com sem sobre entre tipo
parte uma uns umas dele dela deles delas mais menos voce voces tambem muito pouco ainda sempre nunca
fale conte diga mostre compare comparar suponha imagine considere parecido parecida parecidos semelhante
semelhantes similar similares comum tem tema sabe sabem fosse tivesse diferenca diferencas curiosidade
surpreenda duvida duvidas ajuda obrigado obrigada valeu listar liste detalhe detalhes isso esse essa
este esta estes estas aquele aquela quanto quantos quantas nao sim talvez depois pode podem posso""".split())

TONS = ((0.9, "com certeza"), (0.7, "provavelmente"), (0.5, "talvez"), (0.0, "com pouca certeza"))


def tom_de_certeza(c):
    return next(t for lim, t in TONS if c >= lim)


def _juntar(itens):
    itens = list(itens)
    if len(itens) <= 1:
        return "".join(itens)
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def _subjuntivo_para_presente(texto):
    """"o gato voasse" vira "o gato voa": o interpretador só conhece o presente."""
    troca = {"fosse": "é", "fossem": "são", "tivesse": "tem", "tivessem": "têm"}

    def uma(m):
        p = m.group(0)
        baixa = p.lower()
        if baixa in troca:
            return troca[baixa]
        if baixa.endswith("assem") or baixa.endswith("essem"):
            return baixa[:-5] + ("am" if baixa.endswith("assem") else "em")
        return baixa[:-4] + ("a" if baixa.endswith("asse") else "e")
    return re.sub(r"\b\w{2,}(?:ass|ess)em?\b|\bfoss(?:e|em)\b|\btivess(?:e|em)\b", uma, texto,
                  flags=re.I)


@recurso
class Naturalidade(Recurso):
    nome = "naturalidade"
    tipos = ("comparar", "semelhantes", "hipotese", "narrar", "lacunas", "curiosidade",
             "social", "listar", "esclarecer")

    def __init__(self, dialogo):
        super().__init__(dialogo)
        self.quadro = None   # (molde da última pergunta, entidade que ela citava)
        self.curiosidades = 0
        self.duvida = None   # (palavra, [candidatas], frase) quando a grafia é ambígua

    # --- entidades ---
    def _mapa(self):
        return self.d.interprete.entidades()

    def _entidade(self, palavras):
        """A entidade conhecida que o texto designa, ou None. Não inventa entidade nova."""
        palavras = palavras if isinstance(palavras, list) else palavras.split()
        nome = self.d.interprete.nome_entidade(palavras)
        return nome if nome and _chave(nome) in self._mapa() else None

    def _citadas(self, texto):
        """Entidades conhecidas citadas no texto, na ordem, com a posição do token."""
        achadas = []
        toks = self.d.interprete.tokens(texto)
        for i, t in enumerate(toks):
            if t == ",":
                continue
            e = self._entidade([t])
            if e and e not in [x[0] for x in achadas]:
                achadas.append((e, i))
        return achadas, toks

    def _palavras_conhecidas(self):
        itp = self.d.interprete
        conhecidas = set(PALAVRAS_COMUNS) | ARTIGOS | NEGACOES
        for chave in itp.lexico():
            conhecidas.update(chave)
        conhecidas.update(itp.propriedades())
        nomes = set(self.base.entidades()) | {f.o for f in self.base.fatos.values()
                                              if f.r in ("tem_propriedade", "não_tem_propriedade")}
        for nome in nomes:
            conhecidas.update(_sem_acento(w) for w in re.findall(r"[A-Za-z][a-z0-9]*|[a-z0-9]+", _sem_marcas(nome)))
        for forma in SUPERFICIE.values():
            conhecidas.update(_sem_acento(forma).split())
        return conhecidas

    # --- reescrita da entrada: tolerância a erros, pronomes e elipse ---
    def reescrever(self, texto):
        avisos = []
        self.duvida = None
        mapa = self._mapa()
        if not mapa:
            return texto, avisos
        itp = self.d.interprete
        toks = itp.tokens(texto)
        # 1. erro de digitação: palavra desconhecida muito parecida com uma entidade
        conhecidas = self._palavras_conhecidas()
        novos = []
        for t in toks:
            k = _chave(t)
            if (t == "," or len(k) < 4 or k in mapa or _sem_acento(t) in conhecidas
                    or not t.isalpha() or self._entidade([t])):
                novos.append(t)
                continue
            perto = [c for c in difflib.get_close_matches(k, list(mapa), n=3, cutoff=0.74)
                     if c[0] == k[0] and abs(len(c) - len(k)) <= 1]
            if len(perto) >= 2 and (difflib.SequenceMatcher(None, k, perto[0]).ratio()
                                    - difflib.SequenceMatcher(None, k, perto[1]).ratio()) < 0.05:
                # duas entidades igualmente prováveis: pergunta em vez de chutar
                self.duvida = (t, [mapa[c] for c in perto[:2]], texto)
                novos.append(t)
                continue
            if perto:
                alvo = mapa[perto[0]]
                avisos.append(f"Entendi '{t}' como {_legivel(alvo)}.")
                novos.append(alvo)
            else:
                novos.append(t)
        toks = novos
        # 2. pronomes: "ele", "ela", "dele" apontam para o assunto mais recente
        recente = next((e for e in self.d.recentes if _chave(e) in mapa), None)
        if recente and not any(self._entidade([t]) for t in toks if t != ","):
            saida = []
            for t in toks:
                baixa = _sem_acento(t)
                if baixa in ("ele", "ela", "eles", "elas"):
                    saida.append(recente)
                elif baixa in ("dele", "dela", "deles", "delas"):
                    saida += ["de", recente]
                else:
                    saida.append(t)
            if saida != toks:
                avisos.append(f"Entendi o pronome como {_legivel(recente)}.")
            toks = saida
        # 3. elipse: "e o tigre?" repete a última pergunta trocando o assunto
        juntas = " ".join(toks)
        m = re.match(r"^e\s+(?:(?:o|a|os|as|um|uma)\s+)?(\S+)$", juntas, re.I)
        if m and self.quadro and self._entidade([m.group(1)]):
            reescrito = self.quadro[0].replace("{E}", self._entidade([m.group(1)]))
            avisos.append(f"Entendi como: {reescrito}")
            return reescrito, avisos
        texto2 = " ".join(toks)
        if texto.rstrip().endswith("?"):
            texto2 += "?"
        # guarda o molde da pergunta quando ela cita exatamente uma entidade
        citadas, ts = self._citadas(texto2)
        if len(citadas) == 1:
            tipo = itp.interpretar(texto2)["tipo"]
            if tipo in ("o_que", "quem", "descrever", "verificar", "por_que", "narrar"):
                e, pos = citadas[0]
                moldes = [("{E}" if i == pos else t) for i, t in enumerate(ts) if t != ","]
                self.quadro = (" ".join(moldes) + ("?" if texto2.endswith("?") else ""), e)
        return (texto2 if avisos else texto), avisos

    # --- intenções ---
    def interpretar(self, itp, bruto, juntas):
        t = juntas
        if self.duvida:
            return {"tipo": "esclarecer"}
        if t.startswith("e ") and not t.startswith(("e se ", "e ai")):
            t = t[2:]
        # social
        if t in ("oi", "ola", "ola tudo bem", "oi tudo bem", "bom dia", "boa tarde", "boa noite",
                 "e ai", "eae", "opa", "hello", "hi"):
            return {"tipo": "social", "assunto": "oi"}
        if t in ("obrigado", "obrigada", "valeu", "brigado", "muito obrigado", "muito obrigada",
                 "obrigado info", "valeu info"):
            return {"tipo": "social", "assunto": "obrigado"}
        if t in ("quem e voce", "quem voce e", "o que voce e", "qual seu nome", "como voce se chama",
                 "se apresente", "apresente se", "quem e o info"):
            return {"tipo": "social", "assunto": "quem"}
        if t in ("ajuda", "help", "comandos", "o que voce faz", "o que voce sabe fazer",
                 "o que voce consegue fazer", "o que posso perguntar", "o que posso fazer",
                 "como voce funciona", "o que da pra fazer"):
            return {"tipo": "social", "assunto": "ajuda"}
        if t in ("tchau", "ate logo", "ate mais", "adeus", "flw", "falou", "sair", "fechar", "encerrar", "fim"):
            return {"tipo": "social", "assunto": "tchau"}
        if t in ("cancelar", "cancela", "deixa pra la", "esquece", "esqueca isso"):
            return {"tipo": "social", "assunto": "cancelar"}
        if t in ("limpar", "reiniciar", "recomecar", "comecar de novo", "zerar", "resetar", "apagar tudo"):
            return {"tipo": "social", "assunto": "reiniciar"}
        if t in ("listar tudo", "liste tudo", "mostrar tudo", "mostre tudo", "ver tudo", "mostra tudo"):
            return {"tipo": "social", "assunto": "listar_tudo"}
        if t in ("quantos fatos voce tem", "quantos fatos tem", "quantos fatos", "quanto voce sabe",
                 "o tamanho da base"):
            return {"tipo": "resumo"}
        if re.match(r"^(apague|apaga|apagar|remova|remove|remover|esqueca|exclua|delete|deleta) (o |a |os |as )?\w+", t):
            return {"tipo": "social", "assunto": "apagar"}
        if t in ("como voce aprende", "como voce pensa", "como voce raciocina", "voce e uma ia",
                 "voce e inteligente", "voce aprende"):
            return {"tipo": "social", "assunto": "aprende"}
        # metacognição e curiosidade
        if t in ("o que voce nao sabe", "o que voce ainda nao sabe", "onde voce tem duvida",
                 "no que voce tem duvida", "quais as suas duvidas", "quais suas duvidas",
                 "quais sao suas duvidas", "o que voce duvida", "o que falta saber",
                 "qual sua confianca", "quanta certeza voce tem", "quao confiante voce esta",
                 "no que voce nao tem certeza", "do que voce tem duvida"):
            return {"tipo": "lacunas"}
        if t in ("me conte uma curiosidade", "conte uma curiosidade", "me surpreenda", "surpreenda me",
                 "diga algo interessante", "me diga algo interessante", "uma curiosidade",
                 "curiosidade", "me conte algo interessante", "o que voce descobriu",
                 "o que voce deduziu"):
            return {"tipo": "curiosidade"}
        # hipótese
        m = re.match(r"^(?:e se|suponha que|suponha|imagine que|imagine|considere que)\s+(.+?)\??$",
                     bruto, re.I)
        if m:
            interno = _subjuntivo_para_presente(m.group(1))
            itn = itp.interpretar(interno)
            if itn["tipo"] == "afirmar":
                return {"tipo": "hipotese", "fatos": itn["fatos"], "texto": interno}
            if itn["tipo"] == "verificar":
                return {"tipo": "hipotese", "fatos": [itn["fato"]], "texto": interno}
            return {"tipo": "hipotese", "fatos": [], "texto": interno}
        # comparação
        m = (re.match(r"^o que (.+?) e (.+?) (?:tem|possuem|compartilham)(?: em)? comum$", t)
             or re.match(r"^(?:compare|comparar|compara) (.+?) (?:e|com|a|ao) (.+)$", t)
             or re.match(r"^(?:qual a |quais as )?diferencas? entre (.+?) e (.+)$", t))
        if m:
            a, b = self._entidade(m.group(1).split()), self._entidade(m.group(2).split())
            if a and b and a != b:
                return {"tipo": "comparar", "a": a, "b": b}
        m = (re.match(r"^(.+?) (?:e|sao) (?:parecid\w+|semelhantes?|similares?) (?:com|a|ao|aos|as|de) (.+)$", t)
             or re.match(r"^(.+?) se parecem? com (.+)$", t))
        if m:
            a, b = self._entidade(m.group(1).split()), self._entidade(m.group(2).split())
            if a and b and a != b:
                return {"tipo": "comparar", "a": a, "b": b, "parecer": True}
        m = (re.match(r"^(?:quem|o que) (?:se parece|e parecido|e semelhante|lembra|e similar) "
                      r"(?:com|a|ao|de) (.+)$", t)
             or re.match(r"^(?:quem|o que) (?:e )?(?:parecido|semelhante|similar) (?:com|a|ao|de) (.+)$", t))
        if m:
            e = self._entidade(m.group(1).split())
            if e:
                return {"tipo": "semelhantes", "e": e}
        # narrativa e lista
        palavras = t.split()
        n = itp._prefixo(palavras, [("me", "fale", "sobre"), ("fale", "sobre"), ("me", "conte", "sobre"),
                                    ("conte", "sobre"), ("me", "fale", "do"), ("me", "fale", "da"),
                                    ("fale", "do"), ("fale", "da"),
                                    ("o", "que", "voce", "sabe", "sobre"),
                                    ("o", "que", "voce", "sabe", "do"),
                                    ("o", "que", "voce", "sabe", "da"),
                                    ("quem", "e"), ("o", "que", "e")])
        if n and palavras[n:]:
            e = self._entidade(palavras[n:])
            if e:
                return {"tipo": "narrar", "e": e}
        n = itp._prefixo(palavras, [("listar",), ("liste",), ("detalhe",), ("detalhes", "de"),
                                    ("detalhes", "do"), ("detalhes", "da"), ("fatos", "de"),
                                    ("fatos", "do"), ("fatos", "da")])
        if n and palavras[n:]:
            e = self._entidade(palavras[n:])
            if e:
                return {"tipo": "listar", "e": e}
        m = re.match(r"^(?:o |a |os |as )?(.+?) (.+?) (?:o |a |os |as )?(?:que|quem)$", t)
        if m:
            e, rel = self._entidade(m.group(1).split()), itp.lexico().get(tuple(m.group(2).split()))
            if e and rel and rel not in ("é_um", "tem_propriedade"):
                return {"tipo": "o_que", "sujeito": e, "relacao": rel}
        # "onde X vive" / "onde vive X"
        m = (re.match(r"^onde (?:o |a |os |as )?(.+?) (?:vive|mora|habita|vivem|moram)$", t)
             or re.match(r"^onde (?:vive|mora|habita|vivem|moram) (?:o |a |os |as )?(.+)$", t)
             or re.match(r"^(?:o |a |os |as )?(.+?) (?:vive|mora|habita|vivem|moram) onde$", t))
        if m:
            e = self._entidade(m.group(1).split())
            if e:
                return {"tipo": "o_que", "sujeito": e, "relacao": "habitat_principal"}
        return None

    # --- execução ---
    def tratar(self, it):
        tipo = it["tipo"]
        if tipo == "esclarecer":
            palavra, alvos, frase = self.duvida
            self.duvida = None
            return self._esclarecer(palavra, alvos, frase)
        if tipo == "social":
            return self._social(it["assunto"])
        if tipo == "listar":
            return self.d._descrever(it["e"])
        if tipo == "narrar":
            self.d._lembrar(it["e"])
            return self._narrar(it["e"])
        if tipo == "comparar":
            self.d._lembrar(it["b"], it["a"])
            return self._comparar(it["a"], it["b"], it.get("parecer"))
        if tipo == "semelhantes":
            self.d._lembrar(it["e"])
            return self._semelhantes(it["e"])
        if tipo == "hipotese":
            return self._hipotese(it)
        if tipo == "lacunas":
            return self._lacunas()
        if tipo == "curiosidade":
            return self._curiosidade()
        return None

    def _esclarecer(self, palavra, alvos, frase):
        a, b = alvos
        resp = yield (f"Não conheço '{palavra}'. Você quis dizer {_legivel(a)} ou {_legivel(b)}?", "livre")
        chave = _chave(resp)
        escolha = next((x for x in alvos if chave and (_chave(x) in chave or chave in _chave(x))), None)
        palavras = set(_sem_acento(resp).split())
        if escolha is None and palavras & {"primeiro", "primeira", "1"}:
            escolha = a
        if escolha is None and palavras & {"segundo", "segunda", "2"}:
            escolha = b
        if escolha is None:
            return ["Não entendi qual dos dois. Repita a pergunta com o nome certo."]
        self.d.ativo = None
        return [f"Certo, {_legivel(escolha)}."] + self.d._livre_interno(
            re.sub(re.escape(palavra), escolha, frase, count=1, flags=re.I))

    # --- conversa social ---
    def _social(self, assunto):
        n = len(self.base.fatos)
        if assunto == "oi":
            if n:
                return [f"Olá! Estou com {n} fato(s) na memória. Pergunte, conte algo novo ou peça "
                        "ajuda para ver o que sei fazer."]
            return ["Olá! Ainda não tenho base. Me conte algo, por exemplo: o gato caça o rato, "
                    "ou carregue um arquivo."]
        if assunto == "tchau":
            return ["Até logo! O que aprendi nesta conversa fica gravado na base; "
                    "para continuar depois, é só carregar o arquivo de novo."]
        if assunto == "cancelar":
            return ["Não há nada para cancelar agora. Pode perguntar ou contar algo novo."]
        if assunto == "reiniciar":
            return ["Para recomeçar do zero, carregue outro arquivo ou reinicie o ambiente de execução. "
                    "Se só quiser voltar atrás, escreva desfazer."]
        if assunto == "listar_tudo":
            return ["Para ver um assunto fato a fato, escreva listar e o nome (por exemplo: listar Gato). "
                    "Para levar tudo de uma vez, escreva exportar tudo."]
        if assunto == "apagar":
            return ["Não apago fatos direto, para você não perder nada sem querer. Para tirar um fato, "
                    "diga que ele não é verdade (por exemplo: o gato não voa). Para voltar atrás na "
                    "última alteração, escreva desfazer."]
        if assunto == "obrigado":
            return ["Por nada. Se alguma resposta minha estiver errada, diga resposta ruim ou corrija "
                    "na hora que eu aprendo com isso."]
        if assunto == "quem":
            return ["Sou o Info, da DataInfo. Guardo o que você me conta como fatos (sujeito, relação, "
                    "objeto), deduzo coisas novas por regras, calculo o quanto confio em cada fato e "
                    "pergunto quando falta informação. Tudo o que concluo eu sei explicar."]
        if assunto == "aprende":
            return ["Aprendo de quatro jeitos. 1) Cada fato tem uma crença que sobe quando você confirma "
                    "e cai quando nega. 2) Procuro padrões nos dados e proponho regras novas. "
                    "3) Quando uma dedução minha é recusada, a regra que a gerou perde confiança. "
                    "4) Ajusto meus próprios parâmetros com as suas respostas (calibração). "
                    "Escreva métricas para ver como estou indo."]
        return ["Posso, por exemplo:",
                "  guardar o que você disser: o gato caça o rato",
                "  responder: quem caça o sapo? / o que o leão come? / onde o leão vive?",
                "  explicar: por que o leão é um animal?",
                "  falar de um assunto: fale sobre o leão",
                "  comparar: o que o gato e o cão têm em comum? / quem se parece com o leão?",
                "  imaginar: e se o gato voa?",
                "  me auditar: o que você não sabe? / audite a base / métricas",
                "  carregar e exportar arquivos: carregar base.csv / exportar pdf",
                "  voltar atrás ou conferir: desfazer / o que mudou? / salvar",
                "  responder com dúvida a uma pergunta minha: provavelmente sim / acho que não / não sei",
                "  ver o Info em ação: demonstração / verificar ambiente",
                "  explorar bases: grafo interativo / compare com outra.txt / nomes parecidos / "
                "depende da regra X / e se a regra X não existisse?",
                "  ajustar a tela: fonte maior / fonte menor / alto contraste; ata em word"]

    # --- narrativa ---
    def _frases_de(self, e, fatos):
        por_rel = defaultdict(list)
        for f in fatos:
            por_rel[f.r].append(f)
        frases = []
        classes = [f.o for f in por_rel.pop("é_um", [])]
        if classes:
            frases.append(f"{_legivel(e)} é da classe {_juntar(_legivel(c) for c in classes)}.")
        props = [f.o for f in por_rel.pop("tem_propriedade", [])]
        if props:
            verbos = [p for p in props if p in VERBOS_PROPRIEDADE]
            coisas = [p for p in props if p not in VERBOS_PROPRIEDADE]
            partes = verbos + ([f"tem {_juntar(_legivel(p) for p in coisas)}"] if coisas else [])
            frases.append(f"{_legivel(e)} {_juntar(partes)}.")
        nega = [f.o for f in por_rel.pop("não_tem_propriedade", [])]
        if nega:
            verbos = [f"não {p}" for p in nega if p in VERBOS_PROPRIEDADE]
            coisas = [p for p in nega if p not in VERBOS_PROPRIEDADE]
            partes = verbos + ([f"não tem {_juntar(_legivel(p) for p in coisas)}"] if coisas else [])
            frases.append(f"{_legivel(e)} {_juntar(partes)}.")
        for r, lista in por_rel.items():
            frases.append(f"{_legivel(e)} {SUPERFICIE.get(r, _legivel(r))} "
                          f"{_juntar(_legivel(f.o) for f in lista)}.")
        return frases

    def _narrar(self, e):
        fatos = self.base.consultar(s=e, minimo=LIMIAR_ACEITE)
        if not fatos:
            return self.d._descrever(e)
        firmes = [f for f in fatos if f.fonte in FONTES_FIRMES]
        dedu = [f for f in fatos if f.fonte not in FONTES_FIRMES]
        linhas = self._frases_de(e, firmes) or [f"Sobre {_legivel(e)} só tenho dedução."]
        if dedu:
            media = sum(f.confianca for f in dedu) / len(dedu)
            linhas.append(f"Além disso, {tom_de_certeza(media)} (média de {media:.0%}) deduzi:")
            linhas += self._frases_de(e, dedu[:8])
            if len(dedu) > 8:
                linhas.append(f"E mais {len(dedu) - 8} dedução(ões).")
        lacuna = self._lacunas_de(e)
        if lacuna:
            linhas.append(lacuna)
        linhas.append(f"Para ver fato a fato, escreva: listar {_legivel(e)}.")
        return linhas

    def _pares(self, e):
        """Colegas de classe de `e`: entidades que compartilham uma classe com ele."""
        b = self.base
        pares = set()
        for c in {f.o for f in b.consultar(s=e, r="é_um", minimo=LIMIAR_ACEITE)}:
            pares |= {f.s for f in b.consultar(r="é_um", o=c, minimo=LIMIAR_ACEITE)}
        pares.discard(e)
        return pares

    def _lacunas_de(self, e):
        b = self.base
        pares = self._pares(e)
        if len(pares) < 2:
            return None
        tem = {f.r for f in b.consultar(s=e, minimo=LIMIAR_ACEITE)}
        contagem = Counter()
        for p in pares:
            for r in {f.r for f in b.consultar(s=p, minimo=LIMIAR_ACEITE)}:
                contagem[r] += 1
        faltam = [r for r, n in contagem.items()
                  if r not in tem and r != "é_um" and not r.startswith("não_")
                  and n >= max(2, len(pares) / 2)]
        if not faltam:
            return None
        return ("Percebi uma lacuna: os outros da mesma classe costumam ter "
                + _juntar(f"'{_legivel(r)}'" for r in sorted(faltam)[:3])
                + f" e eu não sei isso de {_legivel(e)}.")

    # --- comparação ---
    def _perfil(self, e):
        return {(f.r, f.o): f for f in self.base.consultar(s=e, minimo=LIMIAR_ACEITE)}

    def _similaridade(self, a, b):
        pa, pb = set(self._perfil(a)), set(self._perfil(b))
        uniao = pa | pb
        return (len(pa & pb) / len(uniao)) if uniao else 0.0

    def _sem_sujeito(self, e, r, o):
        return frase_fato(e, r, o).replace(_legivel(e) + " ", "", 1)

    def _comparar(self, a, b, parecer=False):
        pa, pb = self._perfil(a), self._perfil(b)
        comum = sorted(set(pa) & set(pb))
        so_a, so_b = sorted(set(pa) - set(pb)), sorted(set(pb) - set(pa))
        sim = self._similaridade(a, b)
        grau = ("muito parecidos" if sim >= 0.6 else "parecidos" if sim >= 0.35 else
                "pouco parecidos" if sim >= 0.15 else "bem diferentes")
        linhas = [f"{_legivel(a)} e {_legivel(b)} são {grau} no que eu sei "
                  f"(sobreposição de {sim:.0%} dos fatos)."]
        if comum:
            linhas.append("Em comum: " + _juntar(self._sem_sujeito(a, r, o) for r, o in comum[:6]) + ".")
        else:
            linhas.append("Não encontrei nada que os dois tenham em comum na base.")
        if not parecer:
            if so_a:
                linhas.append(f"Só {_legivel(a)}: "
                              + _juntar(self._sem_sujeito(a, r, o) for r, o in so_a[:4]) + ".")
            if so_b:
                linhas.append(f"Só {_legivel(b)}: "
                              + _juntar(self._sem_sujeito(b, r, o) for r, o in so_b[:4]) + ".")
        emb = getattr(self.ia, "embeddings", None)
        if emb is not None and a in emb.ents and b in emb.ents:
            va, vb = emb.E[emb.ents[a]], emb.E[emb.ents[b]]
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
            linhas.append(f"Pela intuição dos embeddings, a proximidade entre eles é {cos:.2f} (de -1 a 1).")
        return linhas

    def _semelhantes(self, e):
        b = self.base
        outros = [x for x in b.entidades() if x != e and b.consultar(s=x, minimo=LIMIAR_ACEITE)]
        ranque = sorted(((self._similaridade(e, x), x) for x in outros), reverse=True)
        ranque = [(s, x) for s, x in ranque if s > 0][:5]
        if not ranque:
            return [f"Não encontrei ninguém parecido com {_legivel(e)} na base."]
        linhas = [f"Quem mais se parece com {_legivel(e)} pelo que sei:"]
        for s, x in ranque:
            comum = sorted(set(self._perfil(e)) & set(self._perfil(x)))
            linhas.append(f"  {_legivel(x)} ({s:.0%}), em comum: "
                          + _juntar(self._sem_sujeito(e, r, o) for r, o in comum[:3]))
        return linhas

    # --- hipótese ---
    def _hipotese(self, it):
        if not it["fatos"]:
            return [f"Não consegui entender a hipótese '{it['texto']}'. Tente no presente, por exemplo: "
                    "e se o gato voa?"]
        clone = copy.deepcopy(self.ia)
        antes = {k for k, f in self.base.fatos.items() if f.confianca >= LIMIAR_ACEITE}
        hip = []
        for s, r, o, neg in it["fatos"]:
            s, r, o, neg = self.d.interprete._efetivo(s, r, o, neg)
            if neg:
                oposta = clone.base.relacao_oposta(r)
                if not oposta:
                    continue
                r = oposta
            clone.base.adicionar(s, r, o, 0.95, "confirmado")
            hip.append((s, r, o))
            self.d._lembrar(s)
        if not hip:
            return ["Essa hipótese é uma negação sem relação oposta declarada, não sei simulá-la."]
        existentes = {c[1:3] for c in self.ia.motor.verificar_consistencia(resolver=False)}
        choques = [c for c in clone.motor.verificar_consistencia(resolver=False)
                   if c[1:3] not in existentes]
        with redirect_stdout(io.StringIO()):
            clone.pensar(verbose=False)
        depois = {k: f.confianca for k, f in clone.base.fatos.items() if f.confianca >= LIMIAR_ACEITE}
        novos = [k for k in depois if k not in antes and k not in hip]
        perdidos = [k for k in antes if k not in depois]
        abalados = [(k, self.base.fatos[k].confianca, depois[k]) for k in antes
                    if k in depois and self.base.fatos[k].confianca - depois[k] >= 0.1]
        linhas = ["Imaginando que " + _juntar(frase_fato(*h) for h in hip)
                  + " (nada disso foi gravado):"]
        for tipo, a, b2, perdedor in choques[:3]:
            linhas.append(f"  Isso contradiz o que sei ({tipo}): {frase_fato(*a)} e {frase_fato(*b2)}.")
            if perdedor:
                linhas.append(f"    Eu teria de descartar: {frase_fato(*perdedor)}, por ter menos confiança.")
        if novos:
            linhas.append(f"  Eu passaria a concluir {len(novos)} coisa(s) nova(s):")
            linhas += [f"    {frase_fato(*k)} ({depois[k]:.0%})" for k in novos[:6]]
            if len(novos) > 6:
                linhas.append(f"    e mais {len(novos) - 6}.")
        if perdidos:
            linhas.append(f"  Deixariam de valer {len(perdidos)} fato(s) por conflito:")
            linhas += [f"    {frase_fato(*k)}" for k in perdidos[:6]]
        if abalados:
            linhas.append(f"  Perderiam confiança {len(abalados)} fato(s):")
            linhas += [f"    {frase_fato(*k)} ({c0:.0%} para {c1:.0%})" for k, c0, c1 in abalados[:6]]
        if not novos and not perdidos and not abalados and not choques:
            linhas.append("  Nada mais mudaria nas minhas conclusões.")
        return linhas

    # --- metacognição ---
    def _lacunas(self):
        b = self.base
        if not b.fatos:
            return ["Ainda não sei nada, então tudo é lacuna. Me conte algo ou carregue um arquivo."]
        dedu = [f for f in b.fatos.values() if f.fonte not in FONTES_FIRMES]
        linhas = []
        if dedu:
            media = sum(f.confianca for f in dedu) / len(dedu)
            linhas.append(f"Das minhas {len(b.fatos)} crenças, {len(dedu)} são deduções minhas. "
                          f"Confio nelas em média {media:.0%}.")
        frageis = sorted((f for f in dedu if LIMIAR_USO <= f.confianca < LIMIAR_ACEITE + 0.2),
                         key=lambda f: f.confianca)[:5]
        if frageis:
            linhas.append("As que mais me deixam em dúvida:")
            linhas += [f"  {frase_fato(f.s, f.r, f.o)} ({f.confianca:.0%})" for f in frageis]
        contagem = {e: len(b.consultar(s=e, minimo=LIMIAR_USO)) for e in b.entidades()}
        pouco = [e for e, n in sorted(contagem.items(), key=lambda x: (x[1], x[0])) if n <= 1][:6]
        if pouco:
            linhas.append("Conheço muito pouco sobre: " + _juntar(_legivel(e) for e in pouco) + ".")
        for e in b.entidades():
            lac = self._lacunas_de(e)
            if lac:
                linhas.append(lac)
                break
        if not linhas:
            linhas.append("Não vejo dúvidas relevantes agora. Tudo o que concluí tem boa sustentação.")
        else:
            linhas.append("Escreva continue para eu perguntar sobre isso.")
        return linhas

    def _curiosidade(self):
        dedu = [f for f in self.base.fatos.values() if f.fonte not in FONTES_FIRMES
                and f.justificativa and f.confianca >= LIMIAR_ACEITE]
        if not dedu:
            return ["Ainda não deduzi nada de interessante. Me conte mais fatos ou regras e eu tento de novo."]
        dedu.sort(key=lambda f: (-len(f.justificativa[1]), -f.confianca, f.chave))
        f = dedu[self.curiosidades % min(len(dedu), 5)]
        self.curiosidades += 1
        self.d._lembrar(f.s)
        return [f"Uma coisa que deduzi sozinho: {frase_fato(f.s, f.r, f.o)} "
                f"({tom_de_certeza(f.confianca)}, {f.confianca:.0%}).",
                f"Cheguei lá porque {self.d._porque_texto(f)}.",
                "Escreva me conte uma curiosidade de novo para outra."]

    # --- última chance de entender ---
    def sem_entendimento(self, texto):
        citadas, _ = self._citadas(texto)
        if not citadas:
            return None
        itp = self.d.interprete
        so_nome = _chave(" ".join(itp._sem_artigos(itp.tokens(texto))))
        if so_nome == _chave(citadas[0][0]):
            return self.d._descrever(citadas[0][0])
        e = _legivel(citadas[0][0])
        return [f"Reconheci {e}, mas não entendi o que você quer saber sobre isso.",
                f"Posso: fale sobre {e} / quem se parece com {e} / e se {e} voa?"]


def _t_contexto_e_elipse():
    d = _dialogo("o leão caça a zebra", "o tigre caça o cervo")
    d.responder("o que o leão caça?")
    r = " ".join(d.responder("e o tigre?"))
    assert "Cervo" in r, r


def _t_tolerancia_a_erro_de_digitacao():
    d = _dialogo("o leão caça a zebra")
    r = " ".join(d.responder("o que o lião caça?"))
    assert "Entendi" in r and "Zebra" in r, r
    assert "Liao" not in d.ia.base.entidades()


def _t_comparar_e_semelhantes():
    d = _dialogo("o gato é um Mamifero", "o cachorro é um Mamifero", "o gato tem pelos",
                 "o cachorro tem pelos", "o gato caça o rato", "o peixe nada")
    r = " ".join(d.responder("o que o gato e o cachorro têm em comum?"))
    assert "pelos" in r.lower() and "Mamifero" in r, r
    r2 = " ".join(d.responder("quem se parece com o gato?"))
    assert "Cachorro" in r2, r2


def _t_hipotese_nao_altera_a_base():
    d = _dialogo("o gato é um Mamifero", "o gato tem pelos")
    antes = set(d.ia.base.fatos)
    r = " ".join(d.responder("e se o gato voasse?"))
    assert "nada disso foi gravado" in r, r
    assert set(d.ia.base.fatos) == antes


def _t_narrativa_e_conversa():
    d = _dialogo("o gato é um Mamifero", "o gato tem pelos", "o gato caça o rato")
    r = " ".join(d.responder("fale sobre o gato"))
    assert "pelos" in r and "caça" in r, r
    assert d.responder("o que você não sabe?")
    assert any("Info" in l for l in d.responder("quem é você?"))


# %% [31] Recursos: transparência (por que não, raciocínio, descartes, avaliação e calibração)
# Quatro formas de o Info mostrar que não é caixa-preta:
#   por que não    "por que o pinguim não é mamífero?": o que bloqueia, e o que faltaria
#   raciocínio     "mostre seu raciocínio": a cadeia de regras em passos numerados
#   descartes      "o que você descartou?": o que saiu da base, quem pediu e quem venceu
#   avaliação      "avalie-se": esconde parte dos fatos e mede quanto recupera (regras e TransE)
#   calibração     "curva de calibração": a confiança que ele declara bate com os acertos?
# A avaliação e a calibração trabalham em cópias: não mexem na base.

def passos_de_prova(base, motor, chave, vistos=None, passos=None):
    """Cadeia de dedução em ordem: primeiro as premissas, depois a conclusão."""
    vistos = set() if vistos is None else vistos
    passos = [] if passos is None else passos
    f = base.fatos.get(chave)
    if f is None or chave in vistos:
        return passos
    vistos.add(chave)
    if f.justificativa and f.fonte not in FONTES_FIRMES:
        nome, premissas = f.justificativa
        for p in premissas:
            passos_de_prova(base, motor, p, vistos, passos)
        regra = motor.regra(nome)
        com = f" (regra '{nome}', {regra.confianca:.0%})" if regra else f" (regra '{nome}')"
        passos.append(f"Como {_juntar(frase_fato(*p) for p in premissas)}{com}, "
                      f"concluo que {frase_fato(*chave)} ({f.confianca:.0%}).")
    else:
        passos.append(f"Parto de que {frase_fato(*chave)} "
                      f"({ROTULO_FONTE.get(f.fonte, f.fonte)}, {f.confianca:.0%}).")
    return passos


def avaliar_ocultos(ia, fracao=0.2, semente=7, minimo=8, epocas=150):
    """Esconde parte dos fatos informados e mede quanto o Info recupera sem eles.

    recuperados: o fato volta por dedução das regras. Hits@k e MRR: o objeto escondido aparece
    entre os k primeiros candidatos do TransE (ranking filtrado, que ignora objetos que já
    formam outro fato conhecido). `acaso` é o MRR de um palpite aleatório, para comparar.
    Devolve None se houver poucos fatos informados.
    """
    firmes = sorted(k for k, f in ia.base.fatos.items()
                    if f.fonte in FONTES_FIRMES and f.confianca >= LIMIAR_ACEITE
                    and k[1] != "não_tem_propriedade")
    if len(firmes) < minimo:
        return None
    ocultos = random.Random(semente).sample(firmes, min(300, max(3, int(len(firmes) * fracao))))
    clone = copy.deepcopy(ia)
    for k in list(clone.base.fatos):
        if k in ocultos or clone.base.fatos[k].fonte not in FONTES_FIRMES:
            clone.base.remover(k)       # tira os escondidos e as deduções antigas
    with redirect_stdout(io.StringIO()):
        clone.pensar(verbose=False)
    recuperados = [k for k in ocultos
                   if k in clone.base.fatos and clone.base.fatos[k].confianca >= LIMIAR_ACEITE]
    def ranquear(emb):
        ranques = []
        for s, r, o in ocultos:
            if s not in emb.ie or o not in emb.ie or r not in emb.ir:
                ranques.append(None)
                continue
            cands = [e for e in emb.ents if e == o or (s, r, e) not in clone.base.fatos]
            dist = {e: emb.distancia(s, r, e) for e in cands}
            ranques.append(1 + sum(1 for e in cands if e != o and dist[e] < dist[o]))
        return ranques

    def metricas(ranques):
        return {"hits1": sum(1 for q in ranques if q == 1) / n,
                "hits3": sum(1 for q in ranques if q is not None and q <= 3) / n,
                "mrr": sum(1 / q for q in ranques if q) / n}

    emb = EmbeddingsTransE(dim=16, epocas=epocas).treinar(clone.base)
    n, total = len(ocultos), len(emb.ents)
    ranques = ranquear(emb)
    rot = metricas(ranquear(EmbeddingsRotatE(dim=16, epocas=epocas).treinar(clone.base)))
    return {"ocultos": ocultos, "n": n, "recuperados": len(recuperados),
            **metricas(ranques), "rotate": rot,
            "acaso": sum(1 / k for k in range(1, total + 1)) / total,
            "entidades": total, "sem_ranque": sum(1 for q in ranques if q is None)}


def dados_calibracao(ia, rotulos, faixas=5):
    """Compara a confiança prevista com o julgamento da pessoa, por faixa de confiança.
    Devolve (lista de (faixa_ini, faixa_fim, n, media_prevista, taxa_real), ECE)."""
    clone = _clone_sem_rotulos(ia, rotulos)
    clone.base.bloqueados = set()
    clone.motor.executar()
    pares = []
    for s, r, o, verdade in rotulos:
        f = clone.base.fatos.get((s, r, o))
        pares.append((f.confianca if f is not None and f.fonte != "observado" else 0.0, verdade))
    linhas, ece = [], 0.0
    for i in range(faixas):
        ini, fim = i / faixas, (i + 1) / faixas
        grupo = [(p, v) for p, v in pares if ini <= p < fim or (i == faixas - 1 and p == 1.0)]
        if not grupo:
            continue
        media = sum(p for p, _ in grupo) / len(grupo)
        real = sum(v for _, v in grupo) / len(grupo)
        linhas.append((ini, fim, len(grupo), media, real))
        ece += len(grupo) / len(pares) * abs(media - real)
    return linhas, ece


@recurso
class Transparencia(Recurso):
    nome = "transparencia"
    tipos = ("porque_nao", "raciocinio", "descartes", "avaliacao", "calibracao")

    # --- reconhecimento ---
    def _fato_da_frase(self, itp, frase):
        itn = itp.interpretar(frase)
        if itn["tipo"] == "afirmar" and itn["fatos"]:
            s, r, o, neg = itn["fatos"][0]
        elif itn["tipo"] == "verificar":
            s, r, o, neg = itn["fato"]
        else:
            return None
        s, r, o, _ = itp._efetivo(s, r, o, False)
        return (s, r, o)

    def interpretar(self, itp, bruto, juntas):
        t = juntas
        m = re.match(r"^por ?que (.+?)\??$", bruto, re.I)
        if m:
            palavras = m.group(1).split()
            if any(_sem_acento(p) == "não" or _sem_acento(p) == "nao" for p in palavras):
                interno = " ".join(p for p in palavras if _sem_acento(p) != "nao")
                fato = self._fato_da_frase(itp, interno)
                if fato:
                    return {"tipo": "porque_nao", "fato": fato}
        m = re.match(r"^(?:prove que|como voce concluiu que|como voce sabe que|"
                     r"como voce chegou a conclusao que|como voce deduziu que)\s+(.+?)\??$", t)
        if m:
            fato = self._fato_da_frase(itp, m.group(1))
            if fato:
                return {"tipo": "raciocinio", "fato": fato}
        if t in ("mostre seu raciocinio", "mostre o raciocinio", "pense em voz alta", "passo a passo",
                 "como voce chegou nisso", "como voce concluiu isso", "explique seu raciocinio",
                 "explique o raciocinio", "mostre o passo a passo", "como voce pensou"):
            return {"tipo": "raciocinio", "fato": None}
        if t in ("o que voce descartou", "o que voce descartou e por que", "o que voce abandonou",
                 "o que voce rejeitou", "o que foi descartado", "o que voce deixou de acreditar",
                 "quais crencas voce descartou", "o que voce jogou fora"):
            return {"tipo": "descartes"}
        if t in ("avalie se", "avalie-se", "avalie se com fatos escondidos", "fatos escondidos",
                 "teste com fatos escondidos", "quanto voce acerta", "teste sua memoria",
                 "quanto voce acerta do que nao viu", "autoavaliacao", "auto avaliacao",
                 "avalie seu desempenho", "se avalie", "avaliacao"):
            return {"tipo": "avaliacao"}
        if t in ("curva de calibracao", "grafico de calibracao", "calibracao", "mostre a calibracao",
                 "mostre a curva de calibracao", "diagrama de confiabilidade", "sua confianca e confiavel"):
            return {"tipo": "calibracao"}
        return None

    def tratar(self, it):
        tipo = it["tipo"]
        if tipo == "porque_nao":
            self.d.ult_fato = it["fato"]
            return self._porque_nao(*it["fato"])
        if tipo == "raciocinio":
            return self._raciocinio(it["fato"] or self.d.ult_fato)
        if tipo == "descartes":
            return self._descartes()
        if tipo == "avaliacao":
            return self._avaliacao()
        return self._calibracao()

    # --- por que não ---
    def _vencedor(self, chave):
        """O fato que derrotou `chave` num conflito, se for possível identificar."""
        b = self.base
        s, r, o = chave
        oposta = b.relacao_oposta(r)
        contra = b.fatos.get((s, oposta, o)) if oposta else None
        if contra is not None and contra.confianca >= LIMIAR_ACEITE:
            return contra
        if r == "é_um":
            for a, c in b.disjuntas:
                if o in (a, c):
                    rival = b.fatos.get((s, "é_um", c if o == a else a))
                    if rival is not None and rival.confianca >= LIMIAR_ACEITE:
                        return rival
        if r in b.funcionais:
            outros = [f for f in b.consultar(s=s, r=r, minimo=LIMIAR_ACEITE) if f.o != o]
            if outros:
                return max(outros, key=lambda f: f.confianca)
        return None

    def _classe_disjunta_de(self, s, classe):
        """True se `classe` (ou uma superclasse dela) é disjunta de alguma classe de `s`."""
        b = self.base

        def acima(c):
            vistos, pilha = {c}, [c]
            while pilha:
                x = pilha.pop()
                for f in b.fatos.values():
                    if f.s == x and f.r == "subclasse_de" and f.o not in vistos:
                        vistos.add(f.o)
                        pilha.append(f.o)
            return vistos

        dele = set()
        for f in b.fatos.values():
            if f.s == s and f.r == "é_um" and f.confianca >= LIMIAR_ACEITE:
                dele |= acima(f.o)
        alvo = acima(classe)
        return any((x in dele and y in alvo) or (y in dele and x in alvo) for x, y in b.disjuntas)

    def _porque_nao(self, s, r, o):
        b, d = self.base, self.d
        chave = (s, r, o)
        f = b.fatos.get(chave)
        if f is not None and f.confianca >= LIMIAR_ACEITE:
            return [f"Na verdade, pelo que sei, {frase_fato(*chave)}: {d._porque_texto(f)}."]
        linhas = [f"Não afirmo que {frase_fato(*chave)} pelos motivos abaixo."]
        if chave in b.rejeitados:
            linhas.append("  Isso foi descartado antes, por você ou por uma verificação.")
        venc = self._vencedor(chave)
        if venc is not None:
            linhas.append(f"  Porque tenho {frase_fato(*venc.chave)} ({venc.confianca:.0%}), "
                          f"que não combina com isso: {d._porque_texto(venc)}.")
            if r == "é_um":
                linhas.append("  As classes são declaradas disjuntas, então uma exclui a outra.")
        if chave in b.bloqueados:
            linhas.append("  Cheguei a deduzir isso, mas a dedução perdeu um conflito para algo "
                          "de maior confiança e foi barrada.")
        if f is not None:
            linhas.append(f"  O indício que tenho é fraco ({f.confianca:.0%}), abaixo do mínimo de "
                          f"{LIMIAR_ACEITE:.0%} para aceitar.")
        classes = ({f.o for f in b.fatos.values() if f.r == "é_um"}
                   | {x for f in b.fatos.values() if f.r == "subclasse_de" for x in (f.s, f.o)})
        faltas = []
        for c, n, falt, _ in self.ia.motor.abduzir(s, r, o, limite=8):
            if falt[1] == "subclasse_de" or (falt[1] == "é_um" and falt[2] not in classes):
                continue    # hipótese sem sentido: classe ao contrário ou objeto que não é classe
            if falt[0] != s and falt[0] in classes:
                continue    # falar de uma classe inteira não responde sobre este indivíduo
            if falt[1] == "é_um" and falt[0] == s and self._classe_disjunta_de(s, falt[2]):
                continue    # exigiria uma classe que exclui as que já conheço
            if self._vencedor(falt) is not None:
                continue    # a premissa que faltaria contradiz algo que já sei
            faltas.append((c, n, falt))
        for c, nome, falt in faltas[:2]:
            linhas.append(f"  Para ser verdade bastaria saber que {frase_fato(*falt)} (regra '{nome}').")
        if len(linhas) == 1:
            linhas.append("  Não tenho nenhum registro nem regra que leve a isso. Na dúvida eu não "
                          "afirmo; ausência de informação não é o mesmo que falso.")
        return linhas

    # --- raciocínio ---
    def _raciocinio(self, chave):
        if not chave:
            return ["Ainda não falamos de nenhum fato. Pergunte algo, por exemplo: "
                    "o gato é um animal? Depois peça: mostre seu raciocínio."]
        f = self.base.fatos.get(chave)
        if f is None or f.confianca < LIMIAR_ACEITE:
            return [f"Não tenho {frase_fato(*chave)} como verdade, então não há raciocínio a mostrar. "
                    "Pergunte por que não, se quiser saber o motivo."]
        passos = passos_de_prova(self.base, self.ia.motor, chave)
        linhas = [f"Meu raciocínio para '{frase_fato(*chave)}' em {len(passos)} passo(s):"]
        linhas += [f"  {i}. {p}" for i, p in enumerate(passos, 1)]
        if f.fonte in FONTES_FIRMES:
            linhas.append("  É um fato informado, não uma dedução.")
        return linhas

    # --- descartes ---
    def _descartes(self):
        b, d = self.base, self.d
        feitos = [e for e in d.diario if e["tipo"] == "descartado"]
        barrados = sorted(b.bloqueados)
        if not feitos and not barrados:
            return ["Ainda não descartei nada nesta conversa."]
        linhas = []
        if feitos:
            linhas.append(f"Você me fez descartar {len(feitos)} coisa(s):")
            for e in feitos[-8:]:
                conf = f", eu confiava em {e['conf']:.0%}" if e.get("conf") else ""
                linhas.append(f"  {e['texto']}{conf}")
        if barrados:
            linhas.append(f"Deduções minhas que perderam conflitos e foram barradas ({len(barrados)}):")
            for k in barrados[:8]:
                venc = self._vencedor(k)
                motivo = (f"perdeu para {frase_fato(*venc.chave)} ({venc.confianca:.0%})"
                          if venc else "perdeu um conflito")
                linhas.append(f"  {frase_fato(*k)}: {motivo}")
        anteriores = len(b.rejeitados) - len(feitos)
        if anteriores > 0:
            linhas.append(f"Há também {anteriores} fato(s) descartado(s) em sessões anteriores.")
        return linhas

    # --- avaliação ---
    def _avaliacao(self):
        res = avaliar_ocultos(self.ia)
        if res is None:
            return ["Preciso de pelo menos 8 fatos informados para esconder alguns e testar. "
                    "Carregue uma base maior ou me conte mais coisas."]
        n = res["n"]
        linhas = [f"Escondi {n} fato(s) que você me informou (nada foi alterado) e tentei recuperá-los.",
                  f"  Por dedução das regras: {res['recuperados']} de {n} "
                  f"({res['recuperados'] / n:.0%}) voltaram sozinhos.",
                  f"  Por intuição dos embeddings (TransE), entre {res['entidades']} candidatos: "
                  f"acerto na 1ª posição {res['hits1']:.0%}, entre os 3 primeiros {res['hits3']:.0%}, "
                  f"MRR {res['mrr']:.2f}.",
                  f"  Com RotatE (que trata relações simétricas): 1ª posição {res['rotate']['hits1']:.0%}, "
                  f"3 primeiros {res['rotate']['hits3']:.0%}, MRR {res['rotate']['mrr']:.2f}.",
                  f"  Para comparar, um palpite aleatório teria MRR {res['acaso']:.2f}."]
        if res["recuperados"] == 0:
            linhas.append("  Nenhum voltou por regra: o que escondi era informação única, que as "
                          "regras não conseguem deduzir de outros fatos. Regras só recuperam o redundante.")
        if res["mrr"] <= res["acaso"] * 1.2:
            linhas.append("  Os embeddings pouco superam o acaso nesta base: ela é pequena demais "
                          "para o TransE aprender padrões. As regras fazem o trabalho.")
        if res["sem_ranque"]:
            linhas.append(f"  {res['sem_ranque']} fato(s) tinham uma entidade que só aparecia neles; "
                          "contei como erro.")
        linhas.append("Resultado de uma amostra só (semente fixa); numa base pequena varia bastante.")
        return linhas

    # --- calibração ---
    def _calibracao(self):
        rotulos = [x for x in self.base.rotulos if len(x) == 4]
        if len(rotulos) < 3:
            return ["Preciso de pelo menos 3 julgamentos seus sobre deduções minhas. Responda "
                    "algumas verificações (Isso procede?) e peça de novo."]
        faixas, ece = dados_calibracao(self.ia, rotulos)
        linhas = [f"Confiança que declaro contra o que você julgou, em {len(rotulos)} caso(s):"]
        for ini, fim, n, media, real in faixas:
            linhas.append(f"  {ini:.0%} a {fim:.0%}: {n} caso(s), eu previa {media:.0%}, "
                          f"você confirmou {real:.0%}")
        linhas.append(f"Erro de calibração esperado (ECE): {ece:.2f} (0 é perfeito).")
        if len(rotulos) < 8:
            linhas.append("São poucos casos; trate o resultado como ilustração, não como medida.")
        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(4.2, 3.6), dpi=130, facecolor="white")
            ax = fig.subplots()
            ax.plot([0, 1], [0, 1], "--", color="#a0aec0", label="calibração perfeita")
            ax.plot([m for _, _, _, m, _ in faixas], [r for _, _, _, _, r in faixas], "o-",
                    color=MARCA["acento_claro"], label="Info")
            ax.set_xlabel("confiança declarada", fontsize=8)
            ax.set_ylabel("fração confirmada", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{MARCA['empresa']} | {MARCA['ia']}: calibração (ECE {ece:.2f})",
                         fontsize=9, color=MARCA["acento_claro"], loc="left")
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            fig.savefig("calibracao.png", bbox_inches="tight")
            linhas.insert(1, "[[img:calibracao.png]]")
        except Exception:
            pass
        return linhas


def _t_esclarece_grafia_ambigua():
    d = _dialogo()
    d.ia.base.adicionar("Tigre", "tem_propriedade", "listras", 1.0, "observado")
    d.ia.base.adicionar("Tigra", "tem_propriedade", "manchas", 1.0, "observado")
    assert "quis dizer" in " ".join(d.responder("o tigro tem manchas?"))
    r = " ".join(d.responder("o segundo"))
    assert "Tigra" in r and "manchas" in r, r


def _t_esclarecimento_nao_perde_a_pergunta():
    d = _dialogo()
    d.ia.base.adicionar("Tigre", "tem_propriedade", "listras", 1.0, "observado")
    d.ia.base.adicionar("Tigra", "tem_propriedade", "manchas", 1.0, "observado")
    d.responder("o tigro voa?")
    d.responder("o primeiro")
    r = " ".join(d.responder("sim"))
    assert "pergunta aberta" not in r, r


def _t_desfazer_preserva_validade():
    d = _dialogo()
    d.ia.base.adicionar("Contrato", "tem_propriedade", "vigente", 1.0, "observado",
                        valido_ate="2030-12-31")
    d.responder("o gato é um animal")
    d.responder("desfazer")
    assert d.ia.base.fatos[("Contrato", "tem_propriedade", "vigente")].valido_ate == "2030-12-31"


def _t_desfazer_cobre_vocabulario():
    d = _dialogo("o gato caça o rato")
    d.ia.base.vocabulario["perseguir"] = "caça"
    d.responder("o rato é um roedor")
    d.ia.base.vocabulario["xyz"] = "caça"
    d.responder("desfazer")
    assert "xyz" not in d.ia.base.vocabulario


def _t_sugestao_usa_a_propria_base_e_avisa_mudancas():
    d = _dialogo("o gato caça o rato")
    r = " ".join(d.responder("xyzzy plugh"))
    assert "Gato" in r or "gato" in r.lower(), r
    d.responder("o rato é um roedor")
    r = " ".join(d.responder("roedor é um tipo de mamífero"))
    assert "mudou" in r, r


def _t_habitat_e_por_que_nao_sem_hipotese_de_classe():
    d = _dialogo("o gato é um felino", "felino é um tipo de mamifero")
    assert "onde Gato vive" in " ".join(d.responder("onde o gato vive?"))
    d.ativo = None
    d.ia.base.disjuntas.append(("Mamifero", "Peixe"))
    d.ia.base.adicionar("Peixe", "subclasse_de", "Animal", 1.0, "observado")
    r = " ".join(d.responder("por que o gato não é um Peixe?"))
    assert "bastaria saber que Gato é da classe Peixe" not in r, r


def _t_comandos_de_cortesia_e_ajuda():
    d = _dialogo("o gato é um mamifero")
    assert "Até logo" in " ".join(d.responder("tchau"))
    assert "desfazer" in " ".join(d.responder("apague o gato"))
    assert "listar" in " ".join(d.responder("listar tudo"))
    assert "Sobre Gato" in " ".join(d.responder("gato"))
    assert "ajuda" in " ".join(d.responder("asdfgh qwerty"))
    assert "desfazer" in " ".join(d.responder("ajuda"))
    d = Dialogo()
    d.responder("o gato é um mamifero")   # abre uma pergunta livre
    assert "deixei essa pergunta de lado" in " ".join(d.responder("cancelar"))
    assert d.ativo is None


def _t_correcao_com_prefixo():
    d = _dialogo("o gato é um mamifero")
    r = " ".join(d.responder("corrija: o gato não é um mamifero"))
    assert "CorrijaOGato" not in r and "Gato" in r, r


def _t_le_tabela_markdown():
    with _pasta_temporaria():
        _escrever("base.md", "# Título\n\nTexto solto que não é fato.\n\n"
                  "| Sujeito | Relação | Objeto |\n|---|---|---|\n"
                  "| Biblioteca | tem_secao | Infantil |\n"
                  "| Biblioteca | é_um | Instituicao |\n\n"
                  "| ID | Termo | Definição |\n|---|---|---|\n| G-001 | Livro | Registro |\n")
        d = Dialogo()
        r = " ".join(d.carregar("base.md"))
        assert "2 fatos" in r, r
        assert ("Biblioteca", "tem_secao", "Infantil") in d.ia.base.fatos


def _t_nome_solto_numa_pergunta_aberta_descreve():
    d = Dialogo()
    d.responder("o gato é um mamifero")
    d.responder("não sei")           # fecha a pergunta de classe
    d.responder("mamifero é um tipo de animal")   # abre "O que você sabe que vale para ... Animal"
    assert d.ativo is not None
    r = " ".join(d.responder("o gato"))
    assert "Sobre Gato" in r and "AnimalO" not in r, r


def _t_rotate_treina_e_ranqueia():
    d = _dialogo("o gato é um mamifero", "o cão é um mamifero", "o leão é um mamifero",
                 "o gato caça o rato", "o cão caça o gato", "o leão caça o cão")
    emb = EmbeddingsRotatE(dim=8, epocas=60).treinar(d.ia.base)
    assert emb.perdas[-1] <= emb.perdas[0]
    assert emb.distancia("Gato", "é_um", "Mamifero") is not None
    assert isinstance(emb.sugerir(d.ia.base, emb.rels, top_k=3), list)


def _t_nomes_parecidos_unem_com_confirmacao():
    d = Dialogo()
    d.responder("chega")
    for s, o in (("Gato", "Mamifero"), ("Cao", "Mamiferos")):
        d.ia.base.adicionar(s, "é_um", o, 1.0, "observado")
    assert nomes_parecidos(d.ia.base) == [("Mamifero", "Mamiferos")]
    r = " ".join(d.responder("nomes parecidos"))
    assert "mesmo que" in r, r
    d.responder("sim")
    assert ("Cao", "é_um", "Mamifero") in d.ia.base.fatos
    assert not any("Mamiferos" in (f.s, f.o) for f in d.ia.base.fatos.values())


def _t_comparar_com_outra_base():
    with _pasta_temporaria():
        _escrever("a.txt", "@funcional tem_capital\nBrasil | tem_capital | Brasilia\nGato | é_um | Mamifero\n")
        _escrever("b.txt", "Brasil | tem_capital | Recife\nGato | é_um | Mamifero\nCao | é_um | Mamifero\n")
        d = Dialogo()
        d.responder("chega")
        d.carregar("a.txt")
        r = " ".join(d.responder("compare com b.txt"))
        assert "Em comum: 1" in r and "Contradições: 1" in r, r


def _t_faq_do_markdown_responde():
    with _pasta_temporaria():
        _escrever("faq.md", "**P-001 O que é o corte?** Data que separa o arquivo do banco.\n"
                  "**P-002 Quais são as filas?** N1, N2 e AT.\n")
        d = Dialogo()
        d.responder("chega")
        d.carregar("faq.md")
        assert len(d.ia.base.faq) == 2
        r = " ".join(d.responder("o que é o corte?"))
        assert "separa o arquivo do banco" in r and "P-001" in r, r
        assert "2 perguntas" in " ".join(d.responder("perguntas frequentes"))


def _t_regra_depende_e_sem_regra():
    d = _dialogo("o gato é um mamifero", "mamifero é um tipo de animal")
    nomes = [r.nome for r in d.ia.motor.regras if r.origem == "estrutural"]
    usada = next(r.nome for r in d.ia.motor.regras
                 if any(f.justificativa and f.justificativa[0] == r.nome for f in d.ia.base.fatos.values()))
    assert "usam a regra" in " ".join(d.responder(f"o que depende da regra {usada}"))
    r = " ".join(d.responder(f"e se a regra {usada} não existisse?"))
    assert "Deixariam de valer" in r and "nada foi alterado" in r, r
    assert "Não conheço a regra" in " ".join(d.responder("sem a regra xyz"))


def _t_demonstracao_passo_a_passo():
    with _pasta_temporaria():
        _escrever("base_animais.txt", "Pinguim | é_um | Ave\nAve | subclasse_de | Animal\nGato | caça | Rato\n")
        d = Dialogo()
        d.responder("chega")
        r = " ".join(d.responder("demonstração"))
        assert "Passo 1 de" in r, r
        r = " ".join(d.responder("continuar"))
        assert "Passo 2 de" in r and "fale sobre o pinguim" in r, r
        assert "Saí do modo" in " ".join(d.responder("parar"))
        assert d.rec["extras"].demo is None


def _t_ambiente_grafo_e_ui():
    with _pasta_temporaria():
        d = _dialogo("o gato caça o rato", "o rato é um roedor")
        assert "Python" in " ".join(d.responder("verificar ambiente"))
        r = d.responder("grafo interativo")
        assert os.path.exists("grafo_info.html") and "[[baixar:grafo_info.html]]" in r
        with open("grafo_info.html", encoding="utf-8") as fp:
            assert "<canvas" in fp.read()
        assert "[[ui:fonte=+1]]" in d.responder("fonte maior")
        assert "[[ui:contraste=1]]" in d.responder("alto contraste")


def _t_ata_em_word():
    try:
        import docx  # noqa: F401
    except ImportError:
        return
    with _pasta_temporaria():
        d = _dialogo("o gato caça o rato")
        r = " ".join(d.responder("ata em word"))
        assert ".docx" in r and any(f.endswith("_ata.docx") for f in os.listdir("."))


def _t_acentos_na_tela_e_no_arquivo():
    with _pasta_temporaria():
        d = _dialogo("o gato é um mamífero")
        assert "Mamífero" in " ".join(d.responder("o gato é um mamífero?"))
        exportar_texto(d.ia, "saida.txt")
        with open("saida.txt", encoding="utf-8") as fp:
            assert "@exibicao Mamifero = Mamífero" in fp.read()


def _t_verificar_nega_quando_ha_conflito():
    d = _dialogo("o plutão é um planeta anão", "o gato é um mamifero")
    d.ia.base.declarar_disjuntas("Mamifero", "PlanetaAnao")
    d.ia.pensar(verbose=False)
    r = " ".join(d.responder("o gato é um planeta anão?"))
    assert r.startswith("Não.") and "não combina" in r, r


def _t_graduada_pesa_menos():
    d = _dialogo()
    assert classificar_resposta("provavelmente sim") == "sim"
    assert classificar_resposta("acho que não") == "nao"
    assert grau_resposta("provavelmente sim") < 1.0 == grau_resposta("sim")
    b = d.ia.base
    b.adicionar("Gato", "é_um", "Mamifero", 1.0, "observado")
    b.adicionar("Mamifero", "subclasse_de", "Animal", 1.0, "observado")
    d.ia.pensar(verbose=False)
    chave = ("Gato", "é_um", "Animal")
    antes = b.fatos[chave].confianca
    d.ia.feedback(*chave, correto=False, imprimir=False, peso=GRAU_PROVAVEL)
    assert chave in b.fatos and b.fatos[chave].confianca < antes   # enfraqueceu, não sumiu


def _t_porque_nao_e_raciocinio():
    d = _dialogo("Mamifero é um tipo de Animal", "o gato é um Mamifero", "o peixe é um Peixe")
    d.ia.base.disjuntas.append(("Mamifero", "Peixe"))
    r = " ".join(d.responder("por que o gato não é um Peixe?"))
    assert "Mamifero" in r and "disjuntas" in r, r
    d.responder("o gato é um animal?")
    r2 = " ".join(d.responder("mostre seu raciocínio"))
    assert "passo" in r2 and "1." in r2, r2


def _t_descartes_e_avaliacao():
    d = _dialogo("o gato tem pelos")
    d._descartar(("Gato", "tem_propriedade", "pelos"))
    assert any("descartar" in l for l in d.responder("o que você descartou?"))
    d2 = _dialogo("Mamifero é um tipo de Animal", *[f"o animal{i} é um Mamifero" for i in range(12)])
    res = avaliar_ocultos(d2.ia)
    assert res and res["n"] >= 3 and 0 <= res["hits3"] <= 1


def _t_calibracao_por_faixa():
    d = _dialogo("Mamifero é um tipo de Animal", "o gato é um Mamifero", "o cao é um Mamifero",
                 "o boi é um Mamifero")
    rot = [("Gato", "é_um", "Animal", 1), ("Cao", "é_um", "Animal", 1), ("Boi", "é_um", "Animal", 0)]
    faixas, ece = dados_calibracao(d.ia, rot)
    assert faixas and 0 <= ece <= 1


# %% [32] Recursos: ferramentas extras (demonstração, ambiente, nomes parecidos, comparar bases, FAQ, regras, grafo)
# Comandos em linguagem natural, todos sem menu:
#   demonstração           o Info executa um roteiro de frases de exemplo, um passo por vez ("continuar")
#   verificar ambiente     confere bibliotecas, arquivos e permissão de escrita, em português
#   nomes parecidos        sugere unir entidades quase iguais (Mamifero e Mamiferos), com confirmação
#   compare com X          diferenças entre a base atual e outro arquivo: só aqui, só lá, contradições
#   perguntas frequentes   respostas documentadas de um Markdown (**P-001 Pergunta?** Resposta)
#   depende da regra X     quais conclusões usam a regra; "e se a regra X não existisse?" simula
#   grafo interativo       grafo_info.html, que abre em qualquer navegador (arrastar, buscar, zoom)
#   fonte maior / menor    ajuste visual do chat; "ata em word" gera a ata em .docx

PASSOS_DEMO = [
    ("fale sobre o pinguim", "Resume o que sabe, com tom de certeza."),
    ("o gato caça o que?", "Uma pergunta simples sobre um fato."),
    ("e ele vive onde?", "Contexto: entende que ele é o gato."),
    ("e o tubarão?", "Elipse: reaproveita a pergunta anterior."),
    ("o gatu é um mamifero?", "Corrige o erro de digitação e avisa o que entendeu."),
    ("o que o gato e o cachorro têm em comum?", "Compara dois assuntos pelos fatos."),
    ("quem se parece com o gato?", "Similaridade entre entidades."),
    ("e se a aguia nada?", "Hipótese: roda numa cópia, nada é gravado."),
    ("e se o pinguim voasse?", "Hipótese que contradiz o que sabe."),
    ("por que a aguia é um animal?", "Prova da dedução, com a regra usada."),
    ("mostre seu raciocínio", "Os passos numerados, com a confiança de cada um."),
    ("por que o pinguim não é um mamífero?", "Explicação contrastiva."),
    ("avalie-se", "Esconde fatos e mede o quanto recupera."),
    ("o que mudou?", "Resumo do que mudou na base nesta sessão."),
]
_ARQUIVOS_DEMO = ("base_animais.txt", os.path.join("exemplos", "base_animais.txt"),
                  os.path.join("bases", "animais.txt"))


def nomes_parecidos(base, minimo=0.9, aproximado=True):
    """Pares de entidades quase iguais: mesma grafia sem acento, plural e, se `aproximado`, uma letra
    de diferença. Nomes que só diferem em dígitos (FilaN1 e FilaN2) e propriedades não contam."""
    props = {f.o for f in base.fatos.values() if f.r in ("tem_propriedade", "não_tem_propriedade")}
    ents = sorted(e for e in base.entidades() if e not in props)
    pares = []
    for i, a in enumerate(ents):
        ka = _chave(a)
        for b in ents[i + 1:]:
            kb = _chave(b)
            if re.sub(r"\d+", "", ka) == re.sub(r"\d+", "", kb) and ka != kb:
                continue
            if ka == kb or (min(len(ka), len(kb)) >= 4 and ka.rstrip("s") == kb.rstrip("s")):
                pares.append((a, b))
            elif (aproximado and min(len(ka), len(kb)) >= 6 and abs(len(ka) - len(kb)) <= 1
                  and difflib.SequenceMatcher(None, ka, kb).ratio() >= minimo):
                pares.append((a, b))
    return pares


def unir_entidades(base, mantido, absorvido):
    """Troca `absorvido` por `mantido` nos fatos firmes; o que era dedução é recalculado depois."""
    movidos = 0
    for chave in [k for k, f in base.fatos.items()
                  if absorvido in (f.s, f.o) and f.fonte in FONTES_FIRMES]:
        f = base.remover(chave)
        s = mantido if f.s == absorvido else f.s
        o = mantido if f.o == absorvido else f.o
        if s == o:
            continue
        novo = base.adicionar(s, f.r, o, 0.5, f.fonte, valido_ate=f.valido_ate)
        novo.alpha, novo.beta = max(novo.alpha, f.alpha), min(novo.beta, f.beta)
        novo.procedencias = (novo.procedencias + f.procedencias)[-6:]
        movidos += 1
    for chave in [k for k, f in base.fatos.items()
                  if absorvido in (f.s, f.o) and f.fonte not in FONTES_FIRMES]:
        base.remover(chave)
    base.rejeitados = {(mantido if s == absorvido else s, r, mantido if o == absorvido else o)
                       for s, r, o in base.rejeitados}
    return movidos


def comparar_bases(a, b):
    """Compara os fatos firmes de duas bases. Devolve (só_a, só_b, em_comum, contradições)."""
    firmes = lambda base: {k for k, f in base.fatos.items()
                           if f.fonte in FONTES_FIRMES and f.confianca >= LIMIAR_ACEITE}
    fa, fb = firmes(a), firmes(b)
    contradicoes = []
    funcionais = a.funcionais | b.funcionais
    disjuntas = list(a.disjuntas) + list(b.disjuntas)
    negacoes = {**a.negacoes, **b.negacoes}
    contrarias = {**negacoes, **{v: k for k, v in negacoes.items()}}
    for s, r, o in sorted(fa):
        if r in funcionais:
            for s2, r2, o2 in fb:
                if (s2, r2) == (s, r) and o2 != o:
                    contradicoes.append(((s, r, o), (s2, r2, o2), "relação funcional: só um valor"))
        if r in contrarias and (s, contrarias[r], o) in fb:
            contradicoes.append(((s, r, o), (s, contrarias[r], o), "afirma e nega"))
        if r == "é_um":
            for s2, r2, o2 in fb:
                if s2 == s and r2 == "é_um" and o2 != o and ((o, o2) in disjuntas or (o2, o) in disjuntas):
                    contradicoes.append(((s, r, o), (s2, r2, o2), "classes disjuntas"))
    return sorted(fa - fb), sorted(fb - fa), sorted(fa & fb), contradicoes


_PARADAS = set("a o as os de do da dos das em no na nos nas e ou um uma uns umas para por com que se "
               "qual quais como quando onde quem eh e_ ao aos".split())


def _palavras_de(texto):
    return {p for p in re.findall(r"[a-z0-9]+", _sem_acento(texto)) if p not in _PARADAS and len(p) > 2}


def faq_do_markdown(texto):
    """Pares (id, pergunta, resposta) de um Markdown: `**P-001 Pergunta?** Resposta` ou
    `**Pergunta?** Resposta` na mesma linha."""
    pares = []
    for linha in texto.splitlines():
        m = re.match(r"^\s*\*\*(?:([A-Za-z]{1,3}-\d+)\s+)?(.+?\?)\*\*\s*(.+)$", linha)
        if m:
            pares.append((m.group(1) or "", m.group(2).strip(), m.group(3).strip()))
    return pares


def melhor_faq(faq, texto, minimo=0.6):
    """O par de FAQ mais parecido com a pergunta (palavras em comum); None se nada passar do mínimo."""
    quer = _palavras_de(texto)
    if not quer:
        return None
    melhor = None
    for i, (_, pergunta, _) in enumerate(faq):
        p = _palavras_de(pergunta)
        if not p:
            continue
        comuns = len(quer & p)
        score = comuns / min(len(quer), len(p))
        if (comuns >= 2 or (comuns == 1 and p == quer)) and score >= minimo \
                and (melhor is None or score > melhor[0]):
            melhor = (score, i)
    return melhor


def gerar_grafo_html(base, caminho, limite=300):
    """Grafo em um único .html, sem biblioteca externa: arrastar nós, zoom, busca, legenda."""
    grau = Counter()
    for f in base.fatos.values():
        if f.confianca >= LIMIAR_ACEITE:
            grau[f.s] += 1
            grau[f.o] += 1
    nos = [n for n, _ in grau.most_common(limite)]
    ids = {n: i for i, n in enumerate(nos)}
    classe = {}
    for f in base.fatos.values():
        if f.r == "é_um" and f.confianca >= LIMIAR_ACEITE and f.s in ids:
            classe.setdefault(f.s, f.o)
    arestas = [{"s": ids[f.s], "t": ids[f.o], "r": SUPERFICIE.get(f.r, _legivel(f.r)),
                "d": f.fonte not in FONTES_FIRMES, "c": round(f.confianca, 2)}
               for f in base.fatos.values()
               if f.confianca >= LIMIAR_ACEITE and f.s in ids and f.o in ids and f.s != f.o]
    dados = {"nos": [{"n": _legivel(n), "g": grau[n], "k": _legivel(classe.get(n, ""))} for n in nos],
             "arestas": arestas, "marca": f"{MARCA['ia']} | {MARCA['empresa']}",
             "cor": MARCA["acento_claro"], "destaque": MARCA["acento_destaque"]}
    html = _HTML_GRAFO.replace("__DADOS__", json.dumps(dados, ensure_ascii=False).replace("</", "<\\/"))
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write(html)
    return len(nos), len(arestas)


_HTML_GRAFO = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Grafo do Info</title>
<style>
:root{--cor:#114984;--dest:#ff7a00;--fundo:#fff;--texto:#1f2328;--suave:#596270;--linha:#d8dde3}
@media(prefers-color-scheme:dark){:root{--fundo:#1b1f24;--texto:#ececec;--suave:#a3a3a3;--linha:#333}}
html,body{margin:0;height:100%;background:var(--fundo);color:var(--texto);
 font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;background:var(--cor);color:#fff;
 border-bottom:3px solid var(--dest)}
header b{font-weight:600}header input{margin-left:auto;padding:6px 10px;border-radius:8px;border:0;min-width:160px}
canvas{display:block;width:100%;height:calc(100% - 52px);cursor:grab}
#dica{position:fixed;left:12px;bottom:12px;font-size:13px;color:var(--suave);background:var(--fundo);
 border:1px solid var(--linha);border-radius:8px;padding:6px 10px;max-width:70%}
</style></head><body>
<header><b id="marca"></b><span id="info"></span><input id="busca" placeholder="Buscar assunto" aria-label="Buscar assunto"></header>
<canvas id="c"></canvas><div id="dica">Arraste os pontos, use a roda para aproximar. Linha tracejada: dedução, aparece ao clicar num ponto. Duplo clique limpa a seleção.</div>
<script>
const D=__DADOS__;
document.getElementById('marca').textContent=D.marca;
const ded=D.arestas.filter(e=>e.d).length;
document.getElementById('info').textContent=D.nos.length+' assuntos, '+(D.arestas.length-ded)+' relações informadas, '+ded+' deduzidas (clique num ponto para vê-las)';
const cv=document.getElementById('c'),cx=cv.getContext('2d');
let W,H,esc=1,ox=0,oy=0,sel=-1,arr=-1,filtro='';
const pal=['#114984','#ff7a00','#2f855a','#805ad5','#c53030','#0f766e','#a16207','#475569'];
const cl={};D.nos.forEach(n=>{if(n.k&&!(n.k in cl))cl[n.k]=Object.keys(cl).length%pal.length});
const N=D.nos.map((n,i)=>({x:Math.cos(i*2.4)*(60+i*3),y:Math.sin(i*2.4)*(60+i*3),vx:0,vy:0,n:n.n,g:n.g,
 c:n.k in cl?pal[cl[n.k]]:'#64748b',r:5+Math.min(10,Math.sqrt(n.g)*2)}));
function ajusta(){W=cv.width=cv.clientWidth*devicePixelRatio;H=cv.height=cv.clientHeight*devicePixelRatio;}
addEventListener('resize',ajusta);ajusta();
function passo(){
 for(let i=0;i<N.length;i++)for(let j=i+1;j<N.length;j++){
  let dx=N[i].x-N[j].x,dy=N[i].y-N[j].y,d2=dx*dx+dy*dy+0.01,f=2500/d2,d=Math.sqrt(d2);
  dx/=d;dy/=d;N[i].vx+=dx*f;N[i].vy+=dy*f;N[j].vx-=dx*f;N[j].vy-=dy*f;}
 D.arestas.forEach(e=>{const a=N[e.s],b=N[e.t];let dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01,
  f=(d-110)*0.02;dx/=d;dy/=d;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;});
 N.forEach((p,i)=>{p.vx-=p.x*0.01;p.vy-=p.y*0.01;const v=Math.hypot(p.vx,p.vy);if(v>15){p.vx*=15/v;p.vy*=15/v;}
  if(i!==arr){p.x+=p.vx*0.5;p.y+=p.vy*0.5;}p.vx*=0.8;p.vy*=0.8;});
}
function casa(p){return !filtro||p.n.toLowerCase().includes(filtro);}
function desenha(){
 cx.setTransform(1,0,0,1,0,0);cx.clearRect(0,0,W,H);
 cx.setTransform(esc*devicePixelRatio,0,0,esc*devicePixelRatio,W/2+ox*devicePixelRatio,H/2+oy*devicePixelRatio);
 const txt=getComputedStyle(document.body).color;
 D.arestas.forEach(e=>{const a=N[e.s],b=N[e.t];const vivo=sel<0||e.s===sel||e.t===sel;
  if(e.d&&!(sel>=0&&vivo))return;
  cx.globalAlpha=vivo?0.55:0.08;cx.strokeStyle=txt;cx.setLineDash(e.d?[4,3]:[]);cx.lineWidth=1;
  cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke();
  if(sel>=0&&vivo){cx.globalAlpha=1;cx.setLineDash([]);cx.fillStyle=txt;cx.font='10px sans-serif';
   cx.fillText(e.r+' ('+Math.round(e.c*100)+'%)',(a.x+b.x)/2,(a.y+b.y)/2);}});
 cx.setLineDash([]);
 N.forEach((p,i)=>{const ok=casa(p);cx.globalAlpha=ok?1:0.12;cx.fillStyle=p.c;cx.beginPath();
  cx.arc(p.x,p.y,p.r,0,7);cx.fill();if(i===sel){cx.lineWidth=2;cx.strokeStyle='#ff7a00';cx.stroke();}
  if(ok&&(p.g>2||i===sel||filtro)){cx.fillStyle=txt;cx.font='11px sans-serif';cx.fillText(p.n,p.x+p.r+3,p.y+3);}});
 cx.globalAlpha=1;
}
function acha(ev){const r=cv.getBoundingClientRect();
 const x=(ev.clientX-r.left-r.width/2-ox)/esc,y=(ev.clientY-r.top-r.height/2-oy)/esc;
 let m=-1,md=1e9;N.forEach((p,i)=>{const d=Math.hypot(p.x-x,p.y-y);if(d<p.r+6&&d<md){md=d;m=i;}});
 return {i:m,x:x,y:y};}
let pan=null;
cv.addEventListener('mousedown',ev=>{const a=acha(ev);if(a.i>=0){arr=a.i;sel=a.i;}else{pan=[ev.clientX-ox,ev.clientY-oy];}});
addEventListener('mousemove',ev=>{if(arr>=0){const a=acha({clientX:ev.clientX,clientY:ev.clientY});
 const r=cv.getBoundingClientRect();N[arr].x=(ev.clientX-r.left-r.width/2-ox)/esc;N[arr].y=(ev.clientY-r.top-r.height/2-oy)/esc;}
 else if(pan){ox=ev.clientX-pan[0];oy=ev.clientY-pan[1];}});
addEventListener('mouseup',()=>{arr=-1;pan=null;});
cv.addEventListener('dblclick',()=>{sel=-1;});
cv.addEventListener('wheel',ev=>{ev.preventDefault();esc*=ev.deltaY<0?1.1:0.9;esc=Math.max(0.2,Math.min(5,esc));},{passive:false});
document.getElementById('busca').addEventListener('input',e=>{filtro=e.target.value.trim().toLowerCase();});
let q=0;
function ajustar(){let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;N.forEach(p=>{x0=Math.min(x0,p.x);x1=Math.max(x1,p.x);y0=Math.min(y0,p.y);y1=Math.max(y1,p.y);});
 const w=cv.clientWidth,h=cv.clientHeight;esc=Math.max(0.2,Math.min(2,0.85*Math.min(w/(x1-x0+80),h/(y1-y0+80))));
 ox=-esc*(x0+x1)/2;oy=-esc*(y0+y1)/2;}
(function ciclo(){passo();if(++q===200)ajustar();desenha();requestAnimationFrame(ciclo);})();
</script></body></html>"""


def _versao(pacote):
    try:
        modulo = __import__(pacote)
        return getattr(modulo, "__version__", "instalado")
    except ImportError:
        return None


@recurso
class Extras(Recurso):
    nome = "extras"
    tipos = ("demo_iniciar", "demo_proximo", "demo_parar", "ambiente", "nomes_parecidos", "comparar_base",
             "faq", "faq_lista", "regra_depende", "regra_sem", "grafo_html", "ui", "ata_docx")

    def __init__(self, dialogo):
        super().__init__(dialogo)
        self.demo = None
        self._pares = []

    # --- reconhecimento ---
    def interpretar(self, itp, bruto, juntas):
        t = juntas.strip(" ?!.")
        if re.match(r"^(modo )?demo(nstracao)?( completa)?$|^(inicie|iniciar|rode|rodar|faca|fazer|comece|comecar) "
                    r"(a |uma )?demo(nstracao)?$", t):
            return {"tipo": "demo_iniciar"}
        if self.demo is not None and t in ("continuar", "continua", "proximo", "seguir", "avancar", "prox", "next"):
            return {"tipo": "demo_proximo"}
        if self.demo is not None and t in ("parar", "pare", "sair da demonstracao", "parar demonstracao",
                                           "encerrar demonstracao", "fim da demonstracao"):
            return {"tipo": "demo_parar"}
        if re.match(r"^(verificar|checar|conferir|testar|diagnosticar) (o )?(meu )?ambiente$|^diagnostico$", t):
            return {"tipo": "ambiente"}
        if re.match(r"^(nomes parecidos|unificar nomes|unir nomes|revisar nomes|nomes duplicados)$", t):
            return {"tipo": "nomes_parecidos"}
        if re.match(r"^(compare|comparar|compara|diferencas|diferenca|o que difere)\b", t):
            alvo = re.search(r"(\S+\.(?:txt|md|csv|json))\b", bruto)
            if alvo:
                return {"tipo": "comparar_base", "arquivo": alvo.group(1).strip("\"'?,;")}
        m = re.match(r"^o que (.+) e$", t)
        if m and "voce" not in m.group(1).split() and len(m.group(1).split()) <= 4:
            toks = itp.tokens(bruto)
            return {"tipo": "descrever", "entidade": itp.nome_entidade(toks[2:-1])}
        if re.match(r"^(perguntas frequentes|faq|quais perguntas voce responde)$", t):
            return {"tipo": "faq_lista"}
        faq = getattr(self.base, "faq", [])
        if faq:
            achado = melhor_faq(faq, bruto, 0.8)
            if achado:
                return {"tipo": "faq", "i": achado[1]}
        m = re.match(r"^(?:e se )?(?:a )?regra (\w+) (?:nao existisse|fosse removida|fosse apagada)$|^sem a regra (\w+)$", t)
        if m:
            return {"tipo": "regra_sem", "regra": m.group(1) or m.group(2)}
        m = re.match(r"^(?:o que|quais conclusoes|quais fatos|o que e que) (?:depende|dependem) d[ao] regra (\w+)$|"
                     r"^(?:onde|quando) (?:se )?usa(?:r)? a regra (\w+)$", t)
        if m:
            return {"tipo": "regra_depende", "regra": m.group(1) or m.group(2)}
        if re.match(r"^(grafo|mostre o grafo|abrir o grafo|exportar o grafo|gerar o grafo)( interativo| em html| html)$|"
                    r"^grafo interativo$|^grafo html$", t):
            return {"tipo": "grafo_html"}
        m = re.match(r"^(fonte|letra) (maior|menor|normal|grande|pequena)$|^(alto contraste|contraste normal)$", t)
        if m:
            return {"tipo": "ui", "acao": m.group(0)}
        if re.match(r"^(gere |gerar |faca |fazer )?(a )?ata (em|no|para) (word|docx)$|^ata docx$|^ata word$", t):
            return {"tipo": "ata_docx"}
        return None

    # --- execução ---
    def tratar(self, it):
        return getattr(self, "_" + it["tipo"])(it)

    # demonstração
    def _demo_iniciar(self, it):
        d = self.d
        if not self.base.fatos or "Pinguim" not in self.base.entidades():
            caminho = next((c for c in _ARQUIVOS_DEMO if os.path.exists(c)), None)
            if caminho is None:
                return ["A demonstração usa a base de animais (base_animais.txt) e não achei esse arquivo. "
                        "Carregue a base de animais e peça a demonstração de novo."]
            self.demo = {"i": -1, "extra": 1}
            return [f"Modo demonstração: vou rodar {len(PASSOS_DEMO) + 1} passos, como se você digitasse cada "
                    "frase. Escreva continuar para avançar ou parar para sair.",
                    f"Passo 1 de {len(PASSOS_DEMO) + 1}: carregar {caminho}",
                    "Mostra: lê a base e aponta o que não bate."] + d.responder(f"carregar {caminho}") + [
                    "Escreva continuar para o próximo passo."]
        self.demo = {"i": -1, "extra": 0}
        return [f"Modo demonstração: {len(PASSOS_DEMO)} passos, como se você digitasse cada frase, sobre a base já "
                "carregada. Escreva continuar para avançar ou parar para sair."]

    def _demo_proximo(self, it):
        self.demo["i"] += 1
        i = self.demo["i"]
        if i >= len(PASSOS_DEMO):
            self.demo = None
            return ["Fim da demonstração. Pode seguir conversando ou pedir desfazer para voltar ao início."]
        frase, comentario = PASSOS_DEMO[i]
        extra = self.demo["extra"]
        saida = self.d.responder(frase)
        return ([f"Passo {i + 1 + extra} de {len(PASSOS_DEMO) + extra}: {frase}", f"Mostra: {comentario}"] + saida
                + ["Escreva continuar para o próximo passo (ou parar)."])

    def _demo_parar(self, it):
        self.demo = None
        return ["Saí do modo demonstração."]

    # ambiente
    def _ambiente(self, it):
        import platform
        linhas = [f"Python {platform.python_version()} em {platform.system()}."]
        faltas = 0
        for pacote, uso in (("numpy", "cálculos"), ("networkx", "grafo"), ("matplotlib", "gráficos"),
                            ("ipywidgets", "chat no notebook"), ("fitz", "PDF (PyMuPDF)"),
                            ("docx", "ata em Word (python-docx)")):
            v = _versao(pacote)
            if v:
                linhas.append(f"  OK: {pacote} {v if v != 'instalado' else ''}".rstrip())
            else:
                faltas += 1
                dica = "instalo sozinho se precisar" if pacote == "fitz" else "só afeta " + uso
                linhas.append(f"  FALTA: {pacote} ({dica})")
        try:
            teste = os.path.join(os.getcwd(), ".teste_info")
            with open(teste, "w", encoding="utf-8") as fp:
                fp.write("ok")
            os.remove(teste)
            linhas.append(f"  OK: consigo gravar arquivos em {os.getcwd()}")
        except OSError as erro:
            faltas += 1
            linhas.append(f"  FALTA: não consigo gravar nesta pasta ({erro.__class__.__name__}). "
                          "Mude de pasta antes de salvar.")
        achados = [c for c in _ARQUIVOS_DEMO if os.path.exists(c)]
        linhas.append(f"  {'OK' if achados else 'AVISO'}: base de exemplo "
                      + (achados[0] if achados else "não encontrada (a demonstração precisa dela)"))
        linhas.append("Tudo pronto." if not faltas else f"{faltas} item(ns) a conferir; o resto funciona.")
        return linhas

    # nomes parecidos
    def _nomes_parecidos(self, it):
        pares = nomes_parecidos(self.base)
        if not pares:
            return ["Não vi nomes parecidos demais na base."]
        self._pares = pares
        return self.d._iniciar(self._q_unir(pares))

    def _q_unir(self, pares):
        total = len(pares)
        acum = [f"Achei {total} par(es) de nomes parecidos. Vou perguntar um de cada vez."]
        for a, b in pares[:5]:
            conta = lambda e: len(self.base.consultar(s=e)) + len(self.base.consultar(o=e))
            mantido, absorvido = (a, b) if conta(a) >= conta(b) else (b, a)
            resp = yield ("\n".join(acum + [f"Tratar {_legivel(absorvido)} como o mesmo que {_legivel(mantido)} "
                                           f"(fica {_legivel(mantido)})?"]), "sim_nao")
            acum = []
            if resp == "sim":
                n = unir_entidades(self.base, mantido, absorvido)
                self.ia.pensar(verbose=False)
                self.d.registrar("decisao", f"uniu {absorvido} em {mantido}")
                acum.append(f"Uni {_legivel(absorvido)} em {_legivel(mantido)} ({n} fato(s) movido(s)).")
            else:
                acum.append(f"Mantive {_legivel(a)} e {_legivel(b)} separados.")
        if total > 5:
            acum.append(f"Ainda restam {total - 5} par(es); escreva nomes parecidos de novo para continuar.")
        return acum or ["Pronto."]

    def ao_carregar(self, info):
        pares = nomes_parecidos(self.base, aproximado=False)
        if not pares:
            return []
        a, b = pares[0]
        return [f"Notei nomes parecidos, como {_legivel(a)} e {_legivel(b)}. Escreva nomes parecidos para revisar."]

    # comparar bases
    def _comparar_base(self, it):
        nome = it.get("arquivo", "")
        candidatos = [nome, os.path.join("bases", nome), nome + ".txt", os.path.join("bases", nome + ".txt")]
        caminho = next((c for c in candidatos if c and os.path.isfile(c)), None)
        if caminho is None:
            return [f"Não encontrei '{nome}'. Escreva o caminho do arquivo, por exemplo: compare com bases/animais.txt"]
        outro = Dialogo(SistemaIA())
        outro.carregar(caminho)
        so_a, so_b, comum, contra = comparar_bases(self.base, outro.ia.base)
        n_a = self.d.arquivo and os.path.basename(self.d.arquivo) or "a base atual"
        n_b = os.path.basename(caminho)
        linhas = [f"Comparei {n_a} com {n_b} (só os fatos informados):",
                  f"  Em comum: {len(comum)}. Só na atual: {len(so_a)}. Só em {n_b}: {len(so_b)}."]
        if contra:
            linhas.append(f"  Contradições: {len(contra)}.")
            for x, y, motivo in contra[:5]:
                linhas.append(f"    {frase_fato(*x)}  x  {frase_fato(*y)} ({motivo})")
        else:
            linhas.append("  Nenhuma contradição entre as duas.")
        for rotulo, lista in ((f"Só na atual", so_a), (f"Só em {n_b}", so_b)):
            if lista:
                linhas.append(f"  {rotulo} (exemplos):")
                linhas += [f"    {frase_fato(*k)}" for k in lista[:4]]
        if so_b:
            linhas.append(f"Para trazer o que só {n_b} tem, escreva carregar {caminho}.")
        return linhas

    # FAQ
    def _faq_lista(self, it):
        faq = getattr(self.base, "faq", [])
        if not faq:
            return ["Não tenho perguntas frequentes carregadas. Carregue um Markdown com **P-001 Pergunta?** Resposta."]
        linhas = [f"Tenho {len(faq)} perguntas frequentes documentadas. Algumas:"]
        linhas += [f"  {p}" for _, p, _ in faq[:6]]
        return linhas

    def _faq(self, it):
        ident, pergunta, resposta = self.base.faq[it["i"]]
        return [resposta, f"(da base de perguntas frequentes{', ' + ident if ident else ''})"]

    def sem_entendimento(self, texto):
        faq = getattr(self.base, "faq", [])
        achado = melhor_faq(faq, texto, 0.55) if faq else None
        if achado:
            return self._faq({"i": achado[1]})
        return None

    # regras
    def _regra(self, nome):
        return next((r for r in self.ia.motor.regras if r.nome == nome), None)

    def _regra_depende(self, it):
        nome = it["regra"]
        if self._regra(nome) is None:
            return [f"Não conheço a regra '{nome}'. Algumas que tenho: "
                    + ", ".join(sorted({r.nome for r in self.ia.motor.regras if r.origem != 'estrutural'})[:8]) + "."]
        usados = [f for f in self.base.fatos.values()
                  if f.fonte not in FONTES_FIRMES and f.justificativa and f.justificativa[0] == nome]
        if not usados:
            return [f"Nenhuma conclusão minha usa a regra '{nome}' agora."]
        linhas = [f"{len(usados)} conclusão(ões) usam a regra '{nome}'. Algumas:"]
        linhas += [f"  {frase_fato(*f.chave)} ({f.confianca:.0%})" for f in usados[:6]]
        return linhas

    def _regra_sem(self, it):
        nome = it["regra"]
        if self._regra(nome) is None:
            return self._regra_depende(it)
        clone = copy.deepcopy(self.ia)
        clone.motor.regras = [r for r in clone.motor.regras if r.nome != nome]
        clone.motor._assinaturas = {r.assinatura(): r for r in clone.motor.regras}
        for k in list(clone.base.fatos):
            if clone.base.fatos[k].fonte not in FONTES_FIRMES:
                clone.base.remover(k)
        with redirect_stdout(io.StringIO()):
            clone.pensar(verbose=False)
        antes = {k for k, f in self.base.fatos.items() if f.confianca >= LIMIAR_ACEITE}
        depois = {k for k, f in clone.base.fatos.items() if f.confianca >= LIMIAR_ACEITE}
        perdidos, novos = sorted(antes - depois), sorted(depois - antes)
        linhas = [f"Imaginando que a regra '{nome}' não existisse (nada foi alterado):"]
        if perdidos:
            linhas.append(f"  Deixariam de valer {len(perdidos)} conclusão(ões):")
            linhas += [f"    {frase_fato(*k)}" for k in perdidos[:6]]
            if len(perdidos) > 6:
                linhas.append(f"    e mais {len(perdidos) - 6}.")
        if novos:
            linhas.append(f"  Passariam a valer {len(novos)} (antes bloqueadas por conflito):")
            linhas += [f"    {frase_fato(*k)}" for k in novos[:4]]
        if not perdidos and not novos:
            linhas.append("  Nada mudaria nas minhas conclusões.")
        return linhas

    # grafo
    def _grafo_html(self, it):
        if not self.base.fatos:
            return ["Ainda não tenho fatos para desenhar. Carregue uma base ou me conte algo."]
        caminho = "grafo_info.html"
        try:
            n, e = gerar_grafo_html(self.base, caminho)
        except OSError as erro:
            return [f"Não consegui gravar o grafo: {erro}"]
        return [f"Gerei o grafo com {n} assuntos e {e} relações. Abra o arquivo no navegador: arraste os "
                "pontos, use a roda para aproximar e a caixa de busca para filtrar.", f"[[baixar:{caminho}]]"]

    # interface
    def _ui(self, it):
        a = it["acao"]
        marcas = {"fonte maior": "fonte=+1", "letra maior": "fonte=+1", "fonte grande": "fonte=+1",
                  "letra grande": "fonte=+1", "fonte menor": "fonte=-1", "letra menor": "fonte=-1",
                  "fonte pequena": "fonte=-1", "letra pequena": "fonte=-1", "fonte normal": "fonte=0",
                  "letra normal": "fonte=0", "alto contraste": "contraste=1", "contraste normal": "contraste=0"}
        return [f"[[ui:{marcas[a]}]]", "Ajustei a aparência do chat. Só vale no notebook; "
                "no terminal, use o zoom do próprio terminal."]

    # ata em word
    def _ata_docx(self, it):
        ata = self.d.rec.get("ata")
        if ata is None:
            return ["A ata não está disponível."]
        return ata.tratar({"tipo": "ata", "formato": "docx"})


# as frases do Extras são específicas; ele é consultado antes dos demais recursos
RECURSOS.insert(0, RECURSOS.pop())


# %% [33] Autoteste: o Info confere a si mesmo
# Cada teste monta uma conversa pequena numa pasta temporária, exercita um recurso e compara
# o resultado com o esperado. "rode os testes" no chat chama executar_testes(). Os mesmos
# testes rodam fora do notebook em tests/test_info.py.

import tempfile
from contextlib import contextmanager


@contextmanager
def _pasta_temporaria():
    anterior = os.getcwd()
    with tempfile.TemporaryDirectory() as pasta:
        os.chdir(pasta)
        try:
            yield pasta
        finally:
            os.chdir(anterior)


def _dialogo(*frases):
    d = Dialogo()
    d.responder("chega")
    for f in frases:
        d.responder(f)
    return d


def _escrever(nome, texto):
    with open(nome, "w", encoding="utf-8") as fp:
        fp.write(texto)


def _t_dedução_basica():
    d = _dialogo("Mamifero é um tipo de Animal", "o gato é um Mamifero")
    assert d.ia.base.fatos[("Gato", "é_um", "Animal")].confianca >= LIMIAR_ACEITE


def _t_negacao():
    d = _dialogo("o gato tem pelos", "o gato não voa")
    assert ("Gato", "não_tem_propriedade", "voa") in d.ia.base.fatos


def _t_conflito_penaliza_uma_vez():
    d = _dialogo()
    b = d.ia.base
    b.declarar_funcional("mora_em")
    b.adicionar("Ana", "mora_em", "Recife", 0.9, "observado")
    b.adicionar("Ana", "mora_em", "Olinda", 0.6, "observado")
    d.ia.pensar(verbose=False)
    c1 = b.fatos[("Ana", "mora_em", "Olinda")].confianca
    for _ in range(3):
        d.ia.pensar(verbose=False)
    assert abs(b.fatos[("Ana", "mora_em", "Olinda")].confianca - c1) < 1e-9


def _t_perdedor_deduzido_fica_bloqueado():
    d = _dialogo()
    b = d.ia.base
    b.declarar_funcional("mora_em")
    b.adicionar("Ana", "mora_em", "Recife", 0.9, "observado")
    d.ia.motor.adicionar_regra(Regra("r", [("?x", "trabalha_em", "?y")], ("?x", "mora_em", "?y"), 0.8))
    b.adicionar("Ana", "trabalha_em", "Olinda", 1.0, "observado")
    for _ in range(3):
        d.ia.pensar(verbose=False)
    assert ("Ana", "mora_em", "Olinda") not in b.fatos or b.fatos[("Ana", "mora_em", "Olinda")].fonte != "inferido"


def _t_ruido_ou_soma_evidencias():
    d = _dialogo()
    b, m = d.ia.base, d.ia.motor
    m.adicionar_regra(Regra("a", [("?x", "p", "?y")], ("?x", "q", "?y"), 0.6))
    m.adicionar_regra(Regra("b", [("?x", "r", "?y")], ("?x", "q", "?y"), 0.6))
    b.adicionar("X", "p", "Y", 1.0, "observado")
    b.adicionar("X", "r", "Y", 1.0, "observado")
    d.ia.pensar(verbose=False)
    melhor = b.fatos[("X", "q", "Y")].confianca
    b.config["acumulo"] = "ruido_ou"
    d.ia.pensar(verbose=False)
    assert b.fatos[("X", "q", "Y")].confianca > melhor + 0.1


def _t_minimo_fuzzy():
    d = _dialogo()
    b, m = d.ia.base, d.ia.motor
    m.adicionar_regra(Regra("a", [("?x", "p", "?y"), ("?x", "r", "?y")], ("?x", "q", "?y"), 0.9))
    b.adicionar("X", "p", "Y", 0.8, "observado")
    b.adicionar("X", "r", "Y", 0.8, "observado")
    d.ia.pensar(verbose=False)
    produto = b.fatos[("X", "q", "Y")].confianca
    d.responder("use lógica fuzzy")
    assert b.fatos[("X", "q", "Y")].confianca > produto + 0.05


def _t_plural_e_singular():
    d = _dialogo("o leão caça o rato")
    d.responder("os leões caçam o gato")
    assert len({f.s for f in d.ia.base.fatos.values() if "Le" in f.s}) == 1


def _t_vocabulario():
    d = _dialogo("o gato caça o rato", "ataca significa caça", "o cachorro ataca o gato")
    assert ("Cachorro", "predador_de", "Gato") in d.ia.base.fatos


def _t_tabela_larga_ida_e_volta():
    with _pasta_temporaria():
        _escrever("a.csv", "Patrimônio,Tipo,Setor\n1,Notebook,TI\n2,Notebook,RH\n3,Monitor,TI\n")
        d = _dialogo()
        d.responder("carregar a.csv")
        d.responder("sim")
        base = {f.chave for f in d.ia.base.fatos.values() if f.fonte in FONTES_FIRMES}
        assert ("1", "setor", "TI") in base and ("3", "é_um", "Monitor") in base
        d.responder("exportar tabela")
        d2 = _dialogo()
        d2.responder("carregar a_tabela.csv")
        d2.responder("sim")
        base2 = {f.chave for f in d2.ia.base.fatos.values() if f.fonte in FONTES_FIRMES}
        assert base == base2, (base ^ base2)


def _t_campos_obrigatorios():
    d = _dialogo("o n1 é um Notebook", "todo notebook precisa de setor e responsável")
    faltas = d.rec["formas"].faltantes()
    assert ("N1", "Notebook", "setor") in faltas
    cand = d.rec["formas"].perguntas(None)
    assert cand and cand[0]["score"] > 1.0


def _t_modelos_sem_valores_inventados():
    d = _dialogo("usar modelo biblioteca")
    b = d.ia.base
    assert "Livro" in b.formas
    inventados = [f for f in b.fatos.values() if f.fonte in FONTES_FIRMES and f.r not in
                  ("subclasse_de",)]
    assert not inventados, inventados


def _t_validade():
    d = _dialogo()
    d.hoje = date(2026, 10, 8)
    d.responder("o n1 é um Notebook até 01/09/2026")
    cand = d.rec["tempo"].perguntas(None)
    assert cand and cand[0]["score"] == 1.0
    d.hoje = date(2026, 8, 15)  # a 17 dias do vencimento: aviso, não vencido
    cand = d.rec["tempo"].perguntas(None)
    assert cand and cand[0]["score"] == 0.7


def _t_procedencia_e_ida_e_volta():
    with _pasta_temporaria():
        _escrever("b.txt", "Gato | é_um | Mamifero | 1.0 | ate=2027-01-31\n")
        d = _dialogo()
        d.responder("carregar b.txt")
        f = d.ia.base.fatos[("Gato", "é_um", "Mamifero")]
        assert f.valido_ate == "2027-01-31" and f.procedencias[0]["arquivo"] == "b.txt"
        d.responder("exportar json")
        d2 = _dialogo()
        d2.responder("carregar b_exportada.json")
        g = d2.ia.base.fatos[("Gato", "é_um", "Mamifero")]
        assert g.valido_ate == "2027-01-31" and g.procedencias


def _t_fonte_perde_peso():
    d = _dialogo()
    b = d.ia.base
    b.registrar_fonte("ruim.csv", False)
    b.registrar_fonte("ruim.csv", False)
    assert b.peso_fonte("ruim.csv") < b.peso_fonte("bom.csv") == 1.0


def _t_calibracao_roda():
    d = _dialogo()
    b = d.ia.base
    b.declarar_transitiva("subclasse_de")
    for i in range(9):
        b.adicionar(f"E{i}", "é_um", "C", 1.0, "observado")
        b.rotulos.append((f"E{i}", "tem_propriedade", "x", i % 2))
    b.adicionar("C", "tem_propriedade", "x", 0.9, "observado")
    d.ia.pensar(verbose=False)
    melhor, antes, depois, _, _ = calibrar(d.ia, b.rotulos, populacao=6, geracoes=3, tempo=5)
    assert depois <= antes + 1e-9


def _t_sessao_e_ata():
    with _pasta_temporaria():
        _escrever("c.txt", "Gato | é_um | Mamifero | 1.0\n")
        d = _dialogo()
        d.responder("carregar c.txt")
        d.responder("o rato é um Mamifero")
        assert os.path.exists(ARQUIVO_SESSAO)
        d.responder("gere a ata")
        assert os.path.exists("c_ata.pdf")
        novo = Dialogo()
        novo.abrir()
        assert novo.ativo is not None and "Na última vez" in novo._pergunta_atual


def _t_motivo_da_pergunta():
    d = Dialogo()
    d.responder("o gato caça o rato")
    d.curioso = True
    d.responder("pode perguntar")
    assert d.ultimo_motivo
    assert "Perguntei porque" in " ".join(d.responder("por que você perguntou isso?"))


TESTES = [(n[3:].replace("_", " "), f) for n, f in sorted(globals().items())
          if n.startswith("_t_") and callable(f)]


def executar_testes(verbose=False):
    """Roda os testes internos. Devolve [(nome, passou, detalhe)]."""
    resultados = []
    for nome, teste in TESTES:
        try:
            with _pasta_temporaria():
                teste()
            resultados.append((nome, True, ""))
        except Exception as erro:
            resultados.append((nome, False, f"{type(erro).__name__}: {erro}"[:200]))
    if verbose:
        for nome, ok, detalhe in resultados:
            print(("ok     " if ok else "FALHOU ") + nome + (f"  {detalhe}" if detalhe else ""))
    return resultados


@recurso
class AutoTeste(Recurso):
    nome = "autoteste"
    tipos = ("testes",)

    def interpretar(self, itp, bruto, juntas):
        if re.match(r"^(?:rode|rodar|execute|executar|faca|fazer)\s+(?:os\s+)?(?:testes|autoteste)$|"
                    r"^autoteste$|^teste-se$|^teste se$|^testes$", juntas):
            return {"tipo": "testes"}
        return None

    def tratar(self, intencao):
        res = executar_testes()
        ok = sum(1 for _, passou, _ in res if passou)
        linhas = [f"Testes: {ok} de {len(res)} passaram."]
        linhas += [f"  falhou: {n} ({detalhe})" for n, passou, detalhe in res if not passou]
        return linhas


# %% [34] Chat: interface de terminal e de notebook
# --- interface com aparência de terminal ---
def _texto_terminal(linha):
    """Fora do notebook, os marcadores de download e de imagem viram o caminho do arquivo."""
    if re.match(r"^\[\[ui:.+\]\]$", linha):
        return None
    m = re.match(r"^\[\[(baixar|img):(.+)\]\]$", linha)
    return f"(arquivo: {m.group(2)})" if m else linha


def prompt_terminal(arquivo=None, ia=None):
    """Mesmo prompt, usando input(): serve para rodar fora de um notebook."""
    dialogo = Dialogo(ia)
    nome = MARCA["ia"]
    print(f"{nome} | {MARCA['empresa']}")

    def falar(linhas):
        # prefixo em texto: o papel de quem fala não depende de cor (nem de NO_COLOR)
        for i, linha in enumerate(x for x in map(_texto_terminal, linhas) if x is not None):
            print((f"{nome}: " if i == 0 else " " * (len(nome) + 2)) + linha)

    falar(dialogo.abrir(arquivo))
    while True:
        try:
            texto = input("Você: ")
        except (EOFError, KeyboardInterrupt):
            break
        if _sem_acento(texto).strip() in ("sair", "exit", "fim"):
            break
        falar(dialogo.responder(texto))
    return dialogo


try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    _TEM_WIDGETS = True
except ImportError:
    _TEM_WIDGETS = False

NOME_IA = MARCA["ia"]

# marca da DataInfo (provisória): um cilindro de dados, o símbolo comum de base de dados
_MARCA_SVG = ("<svg width='__T__' height='__T__' viewBox='0 0 24 24' fill='none' "
              "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
              "stroke-linejoin='round'><ellipse cx='12' cy='6' rx='7' ry='2.8'/>"
              "<path d='M5 6v6c0 1.55 3.13 2.8 7 2.8s7-1.25 7-2.8V6'/>"
              "<path d='M5 12v6c0 1.55 3.13 2.8 7 2.8s7-1.25 7-2.8v-6'/></svg>")


def _logo(tamanho):
    """Logotipo da marca no tamanho pedido: o arquivo de marca.json, ou o cilindro padrão."""
    caminho = MARCA.get("logo_svg")
    if caminho:
        try:
            with open(caminho, encoding="utf-8") as fp:
                svg = fp.read()
            if svg.lstrip().startswith(("<svg", "<?xml")) and "<script" not in svg.lower():
                return (f"<span class='logo-arq' style='display:inline-flex;width:{tamanho}px;"
                        f"height:{tamanho}px'>{svg}</span>")
        except OSError:
            pass
    return _MARCA_SVG.replace("__T__", str(tamanho))

_JS_HISTORICO = """<script>
(function(){
 if(window.__infoHistorico){return;} window.__infoHistorico=true;
 var hist=[],i=0;
 document.addEventListener('keydown',function(e){
  try{
   var el=e.target; if(!el||!el.closest||!el.closest('.ia-entrada')){return;}
   if(e.key==='Enter'){ if(el.value&&el.value.trim()){hist.push(el.value);i=hist.length;} return; }
   if((e.key==='ArrowUp'||e.key==='ArrowDown')&&hist.length){
    i+=(e.key==='ArrowUp'?-1:1); i=Math.max(0,Math.min(hist.length,i));
    el.value=(i<hist.length?hist[i]:''); el.dispatchEvent(new Event('input',{bubbles:true})); e.preventDefault();
   }
  }catch(err){}
 },true);
})();
</script>"""

_CSS_PROMPT = """<style>
.ia-app {
  --fundo:#ffffff; --texto:#1f2328; --suave:#596270; --linha:#e5e7eb; --borda-campo:#8a919c;
  --eu:#e8f0f9; --acento:__CLARO__; --destaque:__DESTAQUE__; --faixa:__CLARO__;
  max-width:780px; margin:0 auto; background:var(--fundo); color:var(--texto);
  border:1px solid var(--linha); border-radius:12px; overflow:hidden;
  font-family:Inter,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
}
@media (prefers-color-scheme: dark) {
  .ia-app { --fundo:#212121; --texto:#ececec; --suave:#a3a3a3; --linha:#333333;
            --eu:#1f344d; --acento:__ESCURO__; --borda-campo:#7a7a7a; --faixa:#0d2f55; }
}
.ia-topo { display:flex; justify-content:space-between; align-items:center;
  padding:12px 20px; font-size:15px; background:var(--faixa); color:#ffffff;
  border-bottom:3px solid var(--destaque); }
.ia-topo .nome { font-weight:600; color:#ffffff; }
.ia-topo .estado { color:#dbe6f3; font-size:12.5px; }
.ia-tela { height:440px; overflow-y:auto; padding:6px 20px 8px;
  display:flex; flex-direction:column-reverse; font-size:15px; line-height:1.7; }
.ia-tela.vazia { flex-direction:column; align-items:center; justify-content:center; }
.ia-tela .vazio { text-align:center; }
.ia-tela .vazio .logo { color:var(--acento); margin-bottom:10px; }
.ia-tela .vazio .titulo { font-size:24px; font-weight:500; margin-bottom:8px; }
.ia-tela .vazio .dica { color:var(--suave); font-size:13.5px; }
.ia-tela .eu { display:flex; justify-content:flex-end; margin:14px 0 10px; }
.ia-tela .eu span { background:var(--eu); padding:9px 16px; border-radius:14px;
  max-width:80%; white-space:pre-wrap; }
.ia-tela .ia { display:flex; gap:14px; margin:12px 0 18px; }
.ia-tela .ia .marca { flex:none; width:28px; height:28px; border-radius:50%;
  background:var(--faixa); border:none; color:#ffffff; display:flex;
  align-items:center; justify-content:center; margin-top:1px; }
.ia-tela .ia .corpo { flex:1; min-width:0; }
.ia-tela .ia p { margin:0 0 6px; white-space:pre-wrap; }
.ia-tela .ia p.rec { padding-left:14px; color:var(--suave); }
.ia-tela a.baixar { color:var(--acento); text-decoration:none;
  border-bottom:1px solid var(--acento); }
.ia-tela img { max-width:100%; border-radius:10px; margin:6px 0; background:#fff;
  border:1px solid var(--linha); }
.ia-barra { margin:6px 18px 8px; padding:8px 10px 8px 8px; background:var(--eu);
  border:1px solid var(--borda-campo);
  border-radius:12px; align-items:center; min-height:52px; }
.ia-barra:focus-within { border-color:var(--destaque); box-shadow:0 0 0 2px var(--destaque); }
.ia-entrada input { background:transparent !important; color:var(--texto) !important;
  border:none !important; box-shadow:none !important; outline:none !important;
  font-size:15px !important; font-family:inherit !important; }
.ia-entrada input::placeholder { color:var(--suave) !important; }
.ia-anexo button { background:transparent !important; border:none !important;
  box-shadow:none !important; color:var(--suave) !important; }
.ia-anexo button:hover { color:var(--texto) !important; }
.ia-enviar button { background:var(--destaque) !important; color:#1f2328 !important;
  border:none !important; border-radius:8px !important; box-shadow:none !important;
  width:36px !important; height:36px !important; padding:0 !important; }
.ia-tela .ia .acoes { margin-top:2px; }
.ia-tela .ia button.copiar { background:transparent; border:none; color:var(--suave);
  cursor:pointer; padding:2px; border-radius:6px; }
.ia-tela .ia button.copiar:hover { color:var(--texto); }
.ia-tela .status { color:var(--suave); font-size:13.5px; margin:8px 0 14px 42px; }
.ia-tela .logo-arq svg { width:100%; height:100%; }
.ia-app.alto { --fundo:#000000; --texto:#ffffff; --suave:#e6e6e6; --linha:#ffffff;
  --eu:#1c1c1c; --acento:#ffd166; --faixa:#000000; --borda-campo:#ffffff; }
.ia-rodape { text-align:center; color:var(--suave); font-size:11.5px; padding:0 20px 14px; }
</style>""".replace("__CLARO__", MARCA["acento_claro"]).replace("__ESCURO__", MARCA["acento_escuro"]).replace("__DESTAQUE__", MARCA["acento_destaque"])


class Prompt:
    """Chat de IA no formato convencional: barra superior com o nome e o estado da base,
    mensagens sem menus, campo de texto arredondado com clipe de anexo e botão de enviar
    (Enter também envia). Tema claro ou escuro conforme o sistema."""

    _cache_arquivos = {}

    def __init__(self, arquivo=None, ia=None):
        if not _TEM_WIDGETS:
            raise RuntimeError("ipywidgets não está disponível; use prompt_terminal().")
        self.dialogo = Dialogo(ia)
        self.turnos = []
        self.escala, self.contraste = 1.0, False
        self.topo = widgets.HTML()
        self.tela = widgets.HTML()
        self.entrada = widgets.Text(placeholder=f"Pergunte ao {MARCA['ia']}",
                                    layout=widgets.Layout(flex="1 1 auto", width="auto"))
        self.entrada.add_class("ia-entrada")
        self.anexo = widgets.FileUpload(accept=".txt,.md,.csv,.json,.pdf,.zip", multiple=False,
                                        description="", icon="paperclip",
                                        layout=widgets.Layout(width="40px"))
        self.anexo.add_class("ia-anexo")
        self.anexo.observe(self._ao_anexar, names="value")
        self.enviar = widgets.Button(description="", icon="arrow-up",
                                     layout=widgets.Layout(width="36px", height="36px"))
        self.enviar.add_class("ia-enviar")
        self.enviar.on_click(lambda _: self._ao_enviar(self.entrada))
        barra = widgets.HBox([self.anexo, self.entrada, self.enviar])
        barra.add_class("ia-barra")
        rodape = widgets.HTML(
            f"<div class='ia-rodape'>{_html.escape(MARCA['ia'])}, da {_html.escape(MARCA['empresa'])}. "
            "As respostas vêm da base carregada; confira as informações importantes.</div>")
        app = widgets.VBox([self.topo, self.tela, barra, rodape])
        app.add_class("ia-app")
        self.app = app
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.entrada.on_submit(self._ao_enviar)
        display(HTML(_CSS_PROMPT))
        display(app)
        display(HTML(_JS_HISTORICO))
        linhas = self.dialogo.abrir(arquivo)
        # sem arquivo e sem base, a tela vazia já orienta; só mostra o que a IA tem a dizer de fato
        self._ia(linhas if (arquivo or self.dialogo.ia.base.fatos) else [])

    # --- desenho ---
    def _cabecalho(self):
        arquivo, fatos = self.dialogo.estado()
        estado = (f"{_html.escape(arquivo)} &middot; {fatos} fatos" if arquivo
                  else (f"{fatos} fatos" if fatos else ""))
        self.topo.value = (f"<div class='ia-topo'><span class='nome' style='display:inline-flex;"
                           f"align-items:center;gap:8px'>{_logo(18)}{_html.escape(NOME_IA)} "
                           f"<span style='font-weight:400;opacity:.85'>| "
                           f"{_html.escape(MARCA['empresa'])}</span></span>"
                           f"<span class='estado'>{estado}</span></div>")

    @classmethod
    def _dados_uri(cls, caminho, mime):
        chave = (caminho, os.path.getmtime(caminho))
        if chave not in cls._cache_arquivos:
            with open(caminho, "rb") as fp:
                cls._cache_arquivos[chave] = (f"data:{mime};base64,"
                                              + base64.b64encode(fp.read()).decode("ascii"))
        return cls._cache_arquivos[chave]

    @classmethod
    def _marcador(cls, tipo, caminho):
        try:
            if os.path.getsize(caminho) > LIMITE_LINK:
                raise OSError("grande demais")
            nome = _html.escape(os.path.basename(caminho))
            if tipo == "img" and not caminho.lower().endswith(".png"):
                raise OSError("só imagens .png")
            if tipo == "img":
                return f"<img src='{cls._dados_uri(caminho, 'image/png')}' alt='{nome}'>"
            return (f"<p><a class='baixar' download='{nome}' "
                    f"href='{cls._dados_uri(caminho, 'application/octet-stream')}'>Baixar {nome}</a></p>")
        except OSError:
            return f"<p class='rec'>Arquivo gravado em {_html.escape(caminho)}</p>"

    @classmethod
    def _bloco_ia(cls, linhas):
        itens = []
        for linha in linhas:
            m = re.match(r"^\[\[(baixar|img):(.+)\]\]$", linha)
            if m:
                itens.append(cls._marcador(m.group(1), m.group(2)))
                continue
            classe = "rec" if linha.startswith("  ") else ""
            itens.append(f"<p class='{classe}'>{_html.escape(linha.strip()) or '&nbsp;'}</p>")
        copiar = ("<div class='acoes'><button class='copiar' title='Copiar resposta' "
                  "aria-label='Copiar resposta' onclick=\"navigator.clipboard && "
                  "navigator.clipboard.writeText(this.closest('.ia').querySelector('.corpo')."
                  "innerText)\"><svg width='15' height='15' viewBox='0 0 24 24' fill='none' "
                  "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
                  "stroke-linejoin='round'><rect x='9' y='9' width='11' height='11' rx='2'/>"
                  "<path d='M5 15V6a2 2 0 0 1 2-2h9'/></svg></button></div>")
        return (f"<div class='ia'><div class='marca'>{_logo(15)}</div>"
                f"<div class='corpo'>{''.join(itens)}{copiar}</div></div>")

    @staticmethod
    def _texto_status(texto):
        """Frase curta sobre o que o Info está fazendo, para operações que demoram."""
        t = _sem_acento(texto)
        for padrao, frase in ((r"^(carreg|abr|ler|leia|import)", "Lendo o arquivo"),
                              (r"^(calibr|otimiz|ajust|evolu|treine)", "Treinando os parâmetros"),
                              (r"padroes|regras|sugir|sugest", "Procurando padrões"),
                              (r"^(export|baix|gere|ata)", "Preparando o arquivo"),
                              (r"teste", "Rodando os testes")):
            if re.search(padrao, t):
                return frase + "..."
        return "Pensando..."

    def _desenhar(self, status=None):
        self._cabecalho()
        if not self.turnos:
            dica = ("Para começar, escreva: carregar base_animais.txt (ou: rode os testes)"
                    if os.path.exists("base_animais.txt")
                    else "Anexe um arquivo de conhecimento com o clipe, ou conte algo para começar.")
            self.tela.value = (f"<div class='ia-tela vazia'><div class='vazio'>"
                               f"<div class='logo'>{_logo(34)}</div>"
                               f"<div class='titulo'>Como posso ajudar?</div>"
                               f"<div class='dica'>{_html.escape(dica)}</div></div></div>")
            return
        corpo = "".join(
            f"<div class='eu'><span>{_html.escape(t)}</span></div>" if quem == "eu"
            else self._bloco_ia(t) for quem, t in self.turnos)
        if status:
            corpo += f"<div class='status'>{_html.escape(status)}</div>"
        self.tela.value = (f"<div class='ia-tela' style='font-size:{15 * self.escala:.1f}px'>"
                           f"<div>{corpo}</div></div>")

    def _eu(self, texto):
        self.turnos.append(("eu", texto))

    def _aplicar_ui(self, linhas):
        """Marcadores [[ui:fonte=+1]] e [[ui:contraste=1]] mudam a tela e não aparecem como texto."""
        restantes = []
        for linha in linhas:
            m = re.match(r"^\[\[ui:(fonte|contraste)=([+-]?\d)\]\]$", linha)
            if not m:
                restantes.append(linha)
            elif m.group(1) == "fonte":
                n = int(m.group(2))
                self.escala = 1.0 if n == 0 else min(1.6, max(0.8, self.escala + 0.15 * n))
            else:
                self.contraste = m.group(2) == "1"
                (self.app.add_class if self.contraste else self.app.remove_class)("alto")
        return restantes

    def _ia(self, linhas):
        linhas = self._aplicar_ui(list(linhas))
        if linhas:
            self.turnos.append(("ia", list(linhas)))
        self._desenhar()

    # --- eventos ---
    def _ao_enviar(self, campo):
        texto = campo.value.strip()
        campo.value = ""
        if not texto:
            return
        self._eu(texto)
        self._desenhar(self._texto_status(texto))
        try:
            resposta = self.dialogo.responder(texto)
        except Exception as erro:
            resposta = [f"Algo deu errado ao processar isso ({type(erro).__name__}). "
                        "Tente reformular a frase; o que já estava na base continua intacto."]
        self._ia(resposta)

    def _ao_anexar(self, mudanca):
        valor = self.anexo.value
        itens = list(valor.values()) if isinstance(valor, dict) else list(valor)
        if not itens:
            return
        for item in itens:
            nome = item.get("name") or item.get("metadata", {}).get("name", "anexo.txt")
            os.makedirs("anexos_info", exist_ok=True)
            base_nome, ext_nome = os.path.splitext(os.path.basename(nome))
            caminho = os.path.join("anexos_info", base_nome + ext_nome)
            n = 0
            while os.path.exists(caminho):
                n += 1
                caminho = os.path.join("anexos_info", f"{base_nome}_{n}{ext_nome}")
            with open(caminho, "wb") as fp:
                fp.write(bytes(item["content"]))
            self._eu(f"Anexei {os.path.basename(nome)}")
            self._desenhar("Lendo o arquivo...")
            try:
                resposta = self.dialogo.anexar(caminho)
            except Exception as erro:
                resposta = [f"Não consegui ler {caminho} ({type(erro).__name__}). "
                            "Confira se é um arquivo de texto, CSV ou JSON válido."]
            self._ia(resposta)
        try:
            self.anexo.value = ()
        except Exception:
            pass


# %% [35] Arquivo de exemplo para o prompt
# Escreve base_animais.txt neste ambiente, se ainda não existir. É uma base de
# zoologia com lacunas e uma contradição de propósito: o prompt a lê, pergunta o que
# falta e grava as novidades em base_animais_atualizada.txt. Para usar outro assunto,
# anexe um arquivo no mesmo formato (sujeito | relação | objeto | confiança).

BASE_EXEMPLO = """# Base de conhecimento de zoologia
# Formato de cada linha:  sujeito | relação | objeto | confiança (opcional, de 0 a 1)
# Linhas que começam com @ declaram propriedades das relações e regras.
# Linhas que começam com # são comentários. Também é aceita uma frase simples, como: o gato caça o rato

# --- propriedades das relações ---
@oposta tem_propriedade não_tem_propriedade
@transitiva subclasse_de
@transitiva evoluiu_de
@inversa predador_de presa_de
@simetrica parceiro_de
@simetrica compete_com
@funcional habitat_principal
@disjuntas Ave Mamifero
@disjuntas Mamifero Peixe
@disjuntas Ave Peixe

# --- regra de conhecimento (código 1 da aula): penas e nada sugerem ave aquática ---
@regra ave_aquatica: (?a tem_propriedade penas) & (?a tem_propriedade nada) => (?a é_um AveAquatica) 0.85

# --- taxonomia ---
Ave | subclasse_de | Animal
Mamifero | subclasse_de | Animal
Peixe | subclasse_de | Animal
Reptil | subclasse_de | Animal
AveAquatica | subclasse_de | Ave
Animal | subclasse_de | SerVivo

# --- o que cada classe tem em comum ---
Animal | tem_propriedade | respira
Ave | tem_propriedade | penas
Ave | tem_propriedade | voa
Mamifero | tem_propriedade | pelos
Mamifero | tem_propriedade | mama
Peixe | tem_propriedade | nada
Peixe | tem_propriedade | escamas
Reptil | tem_propriedade | escamas

# --- aves ---
Aguia | é_um | Ave
Pato | é_um | Ave
Pato | tem_propriedade | nada
Cisne | é_um | Ave
Cisne | tem_propriedade | nada
Pinguim | é_um | Ave
Pinguim | tem_propriedade | nada
Pinguim | não_tem_propriedade | voa
Pata | é_um | Ave
Pato | parceiro_de | Pata

# --- mamíferos ---
Gato | é_um | Mamifero
Cachorro | é_um | Mamifero
Lobo | é_um | Mamifero
Golfinho | é_um | Mamifero
Golfinho | tem_propriedade | nada
Baleia | é_um | Mamifero
Baleia | tem_propriedade | nada

# --- peixes e répteis ---
Tubarao | é_um | Peixe
Salmao | é_um | Peixe
Cobra | é_um | Reptil
Cobra | não_tem_propriedade | pelos

# --- cadeia alimentar ---
Aguia | predador_de | Salmao
Tubarao | predador_de | Salmao
Golfinho | predador_de | Salmao
Gato | predador_de | Rato
Cobra | predador_de | Sapo
Tubarao | predador_de | Golfinho | 0.5

# --- competição por alimento (a IA pode descobrir que predadores da mesma presa competem) ---
Aguia | compete_com | Tubarao
Tubarao | compete_com | Golfinho

# --- evolução ---
Cachorro | evoluiu_de | Lobo
Lobo | evoluiu_de | Canideo
Canideo | evoluiu_de | Carnivoro

# --- habitat (de propósito, há um conflito e várias lacunas) ---
Golfinho | habitat_principal | Oceano
Golfinho | habitat_principal | RioDoce | 0.6
Tubarao | habitat_principal | Oceano
Salmao | habitat_principal | Rio
Pinguim | habitat_principal | Antartida

# --- entidades sem classe, para a IA perguntar ---
Rato | tem_propriedade | pelos
Sapo | tem_propriedade | respira
"""

if not os.path.exists("base_animais.txt"):
    with open("base_animais.txt", "w", encoding="utf-8") as _arq:
        _arq.write(BASE_EXEMPLO)
    print("Escrevi base_animais.txt neste ambiente.")


# %% [36] Abrir o prompt
# Escreva frases comuns. Para carregar uma base, escreva: carregar base_animais.txt
# ou use o clipe de anexo ao lado da linha de texto.

if _TEM_WIDGETS:
    prompt = Prompt()
else:
    prompt = None
    print("ipywidgets não encontrado: use prompt_terminal().")
