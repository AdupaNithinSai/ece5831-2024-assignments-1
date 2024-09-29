# logic_gate.py
import numpy as np
import sys

class logicgate:
    def and_gate(x1, x2):
        return np.logical_and(x1, x2).astype(int)

    @staticmethod
    def nand_gate(x1, x2):
        return np.logical_not(np.logical_and(x1, x2)).astype(int)

    @staticmethod
    def or_gate(x1, x2):
        return np.logical_or(x1, x2).astype(int)

    @staticmethod
    def nor_gate(x1, x2):
        return np.logical_not(np.logical_or(x1, x2)).astype(int)

    @staticmethod
    def xor_gate(x1, x2):
        return np.logical_xor(x1, x2).astype(int)


if __name__ == "__main__":
    print("LogicGate Class Help:")
    print("This class implements logic gates using numpy.")
    print("Methods available:")
    print("  - and_gate(input1, input2): AND gate")
    print("  - nand_gate(input1, input2): NAND gate")
    print("  - or_gate(input1, input2): OR gate")
    print("  - nor_gate(input1, input2): NOR gate")
    print("  - xor_gate(input1, input2): XOR gate")
    print("\nUsage example in another script:")
    print("    from logic_gate import LogicGate")
    print("    result = LogicGate.and_gate(input1, input2)")