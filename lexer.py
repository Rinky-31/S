from re import match


class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __str__(self):
        return f"{self.__class__.__name__}({self.type}, {self.value})"

    def __repr__(self):
        return str(self)


TOKENS_REPLACE: list[tuple[str, str]] = [
    ("FLOAT_NUMBER", r"\d+\.\d+"),
    ("NUMBER", r"\d+"),
    ("ENDEXPR", r";"),
    ("ENDL", r"\n"),
    ("INCREMENT", r"\+\+"),
    ("DECREMENT", r"\-\-"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("PLUS", r"\+"),
    ("MINUS", r"\-"),
    ("SLASH", r"\/"),
    ("STAR", r"\*"),
    ("AND", r"\band\b|\bі\b"),
    ("OR", r"\bor\b|\bабо\b"),
    ("OF", r"\bof\b|\bз\b"),
    ("CONTAINS", r"\bcontains\b|\bмістить\b"),
    ("SPREAD", r"\.\.\."),
    ("GREATER_EQ", r">="),
    ("LESS_EQ", r"<="),
    ("GREATER", r">"),
    ("LESS", r"<"),
    ("EQUALS_CHECK", r"=="),
    ("NOT_EQUALS_CHECK", r"!="),
    ("EQUALS", r"="),
    ("STRING_LITERAL", r"([\"'])(?:\\.|(?!\1).)*\1"),
    (
        "KEYWORD",
        r"\b(loop|for|while|func|class|drop|ret|load|if|elif|else|switch|case|another|break|continue|delete|req|цикл|для|поки|функція|клас|викинути|повернути|завантажити|якщо|інакшеякщо|інакше|зупинити|пропустити|видалити|необхідний)\b",
    ),
    ("DOUBLE_DOT", r":"),
    ("COMMENT", r"\\\\"),
    ("SPACE", r"\s+"),
    ("NAME", r"\w+"),
    ("COMMA", r","),
    ("LBRACKET", r"{"),
    ("RBRACKET", r"}"),
    ("LSQBRACKET", r"\["),
    ("RSQBRACKET", r"\]"),
]

TRANSLATED_TOKENS_NAME = {
    "клас": "class",
    "цикл": "loop",
    "для": "for",
    "поки": "while",
    "функція": "func",
    "викинути": "drop",
    "перечислення": "enumerate",
    "повернути": "ret",
    "завантажити": "load",
    "якщо": "if",
    "інакшеякщо": "elif",
    "інакше": "else",
    "брехня": "false",
    "правда": "true",
    "зупинити": "break",
    "пропустити": "continue",
    "видалити": "delete",
    "необхідний": "req",
    "і": "and",
    "або": "or",
    "з": "of",
    "містить": "contains",
    "конструктор": "ctor",
    "деструктор": "dtor",
    "цей": "this",
}


def get_tokens(code: str) -> list[Token]:
    tokens: list[Token] = []
    while code:
        for type, regex in TOKENS_REPLACE:
            if res := match(regex, code):
                res = res.group(0)
                if type != "SPACE":
                    tokens.append(Token(type, TRANSLATED_TOKENS_NAME.get(res, res)))
                code = code[len(res) :]
                break
        else:
            print(code[0])
            raise SyntaxError(f"Unexcepted token: {code[0]}")
    return tokens
