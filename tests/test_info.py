"""Testes do Info. Rodar a partir da pasta do projeto:  python -m unittest discover -s tests -v

Cada teste interno de executar_testes() vira um caso separado, então a falha aponta o recurso.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import info_datainfo as info  # noqa: E402


class TestesDoInfo(unittest.TestCase):
    pass


def _caso(nome, funcao):
    def teste(self):
        anterior = os.getcwd()
        with info._pasta_temporaria():
            funcao()
        self.assertEqual(os.getcwd(), anterior)
    return teste


for _nome, _funcao in info.TESTES:
    setattr(TestesDoInfo, "test_" + _nome.replace(" ", "_"), _caso(_nome, _funcao))


class TestesDeFormato(unittest.TestCase):
    def test_modulo_importa_sem_efeitos(self):
        self.assertTrue(hasattr(info, "Dialogo"))
        self.assertTrue(info.RECURSOS)


if __name__ == "__main__":
    unittest.main()
