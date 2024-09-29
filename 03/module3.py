# module3.py
from logic_gate import logicgate
import numpy as np

def test_logic_gates():
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])

    print("AND Gate:")
    print(logicgate.and_gate(x1, x2))

    print("NAND Gate:")
    print(logicgate.nand_gate(x1, x2))

    print("OR Gate:")
    print(logicgate.or_gate(x1, x2))

    print("NOR Gate:")
    print(logicgate.nor_gate(x1, x2))

    print("XOR Gate:")
    print(logicgate.xor_gate(x1, x2))

if __name__ == "__main__":
    test_logic_gates()
