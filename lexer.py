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
    ("AND", r"and"),
    ("OR", r"or"),
    ("OF", r"of"),
    ("CONTAINS", r"contains"),
    ("SPREAD", r"\.\.\."),
    ("GREATER_EQ", r">="),
    ("LESS_EQ", r"<="),
    ("GREATER", r">"),
    ("LESS", r"<"),
    ("EQUALS_CHECK", r"=="),
    ("EQUALS", r"="),
    ("STRING_LITERAL", r"\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^\'\\])*\'"),
    ("KEYWORD", r"(loop|for|while|func|drop|constant|enumerate|ret|load|if|elif|else|false|true|break|continue|delete|req)"),
    ("DOUBLE_DOT", r":"),
    ("COMMENT", r"\/\-"),
    ("SPACE", r"\s+"),
    ("NAME", r"\w+"),
    ("COMMA", r","),
    ("LBRACKET", r"{"),
    ("RBRACKET", r"}"),
]

    

def get_tokens(code: str) -> list[Token]:
    tokens: list[Token] = []
    while code:
        for type, regex in TOKENS_REPLACE:
            if res := match(regex, code):
                res = res.group(0)
                if type != "SPACE":
                    tokens.append(Token(type, res))
                code = code[len(res):]
                break
        else:
            raise SyntaxError(f"Unexcepted token: {code[0]}")
    return tokens