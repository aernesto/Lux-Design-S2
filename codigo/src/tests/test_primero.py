# -*- coding: utf-8 -*-
def funcion_a(lista):
    return len([a for a in lista if a == 2])


def test_function_a():
    prueba = [2, 3, 4, 5, 5, 5, 1, 2, 2, 2]
    assert funcion_a(prueba) == 4


if __name__ == "__main__":
    pass
