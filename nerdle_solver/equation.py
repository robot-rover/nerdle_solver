import string


def apply_operator(operator, left, right):
    if operator == '*':
        return left * right
    elif operator == '/':
        return left // right
    elif operator == '+':
        return left + right
    elif operator == '-':
        return left - right
    raise RuntimeError(f"Invalid Operator {operator}")


def parse(text):
    chars = text.replace(' ', '')
    tokens = []
    for char in chars:
        if char in string.digits:
            if len(tokens) > 0 and isinstance(tokens[-1], int):
                tokens[-1] = tokens[-1] * 10 + int(char)
            else:
                tokens.append(int(char))
        elif char in '+-/*=':
            tokens.append(char)
        else:
            raise RuntimeError(f'Invalid character in equation: "{char}"')
    return tokens


def evaluate(tokens):
    order = ['*/', '+-']
    for level in order:
        idx = 0
        while idx < len(tokens):
            operator = tokens[idx]
            if isinstance(operator, str) and operator in level:
                if idx < 1 or not isinstance(tokens[idx - 1], int):
                    return None
                left = tokens[idx - 1]
                if idx >= len(tokens) - 1 or not isinstance(tokens[idx + 1], int):
                    return None
                right = tokens[idx + 1]
                tokens = tokens[:idx - 1] + tokens[idx + 1:]
                if operator == '/':
                    if right == 0:
                        return None
                    if (left // right) * right != left:
                        return None
                tokens[idx - 1] = apply_operator(operator, left, right)
            else:
                idx += 1

    if len(tokens) > 1:
        return None
    return tokens[0]


def validate(tokens):
    if len(tokens) < 3:
        return False
    result = tokens[-1]
    if not isinstance(result, int):
        return False
    if tokens[-2] != '=':
        return False

    return evaluate(tokens[:-2]) == result
