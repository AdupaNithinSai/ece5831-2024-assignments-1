<<<<<<< HEAD
# pricing.py

def get_net_price(price, tax_rate, discount=0):
    return price * (1 + tax_rate) * (1-discount)


def get_tax(price, tax_rate=0):
=======
# pricing.py

def get_net_price(price, tax_rate, discount=0):
    return price * (1 + tax_rate) * (1-discount)


def get_tax(price, tax_rate=0):
>>>>>>> e179c21965b3529d6fd0695c67f1567028c0c5a0
    return price * tax_rate