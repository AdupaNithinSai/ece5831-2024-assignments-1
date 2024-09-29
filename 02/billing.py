<<<<<<< HEAD
def calculate_tax(price, tax):
    return price * tax


def print_billing_doc():
    tax_rate = 0.1
    products = [{'name': 'Book',  'price': 30},
                {'name': 'Pen', 'price': 5}]

    # print billing header
    print(f'Name\tPrice\tTax')

    # print the billing item
    for product in products:
        tax = calculate_tax(product['price'], tax_rate)
        print(f"{product['name']}\t{product['price']}\t{tax}")


# print(__name__)

if __name__ == '__main__':
=======
def calculate_tax(price, tax):
    return price * tax


def print_billing_doc():
    tax_rate = 0.1
    products = [{'name': 'Book',  'price': 30},
                {'name': 'Pen', 'price': 5}]

    # print billing header
    print(f'Name\tPrice\tTax')

    # print the billing item
    for product in products:
        tax = calculate_tax(product['price'], tax_rate)
        print(f"{product['name']}\t{product['price']}\t{tax}")


# print(__name__)

if __name__ == '__main__':
>>>>>>> e179c21965b3529d6fd0695c67f1567028c0c5a0
    print_billing_doc()