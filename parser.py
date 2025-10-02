from lexer import Token, get_tokens
from typing import Optional




class Instruction:
    def __init__(self, type: str, value=None):
        self.type = type
        self.value = value

class IF:
    def __init__(self, ifs: Optional["Node"] = None, Elifs: Optional[list["Node"]] = None, Else: Optional["Node"] = None):
        self.If = ifs
        self.Elifs = Elifs or []
        self.Else = Else
    
    @staticmethod
    def from_list(lst: list["Node"]):
        res = IF()
        for i in lst:
            match i.type:
                case "_IF":
                    res.If = i
                case "_ELIF":
                    res.Elifs.append(i)
                case "_ELSE":
                    res.Else = i
        return res

class Environment:
    def __init__(
        self, vars: Optional[dict[str]] = None, parent: Optional["Environment"] = None
    ):
        self.vars = vars or {}
        self.parent = parent

    def set(self, varname: str, value):
        self.vars[varname] = value

    def get(self, varname: str, env: Optional["Environment"] = None):
        env = env or self
        res = env.vars.get(varname)
        if res is None and env.parent:
            res = self.get(varname, env.parent)
        return res


class Function:
    def __init__(
        self,
        argcount: int = 0,
        body: list["Node"] = [],
        params_name: Optional[list] = None,
        _env: Optional[Environment] = None,
        *,
        executor=None,
        name: str = None,
        global_executor_env: bool = False,
    ):
        self.argcount = argcount
        self.body = list(filter(None, body))
        self.executor = executor
        self.env = Environment(parent=_env or env)
        self.is_global_executor = global_executor_env
        self.params_name = params_name or []
        self.name = name

    def exec_body(self, args: list):
        args = list(filter(lambda a: a is not None, args))
        local_env = Environment(parent=self.env)
        local_env.vars.update(zip(self.params_name, args))
        if self.name:
            local_env.set(self.name, self)
        if self.argcount != len(args) and self.argcount != -1:
            raise TypeError(
                f"Function get {len(args)} params, but {self.argcount} excepted"
            )
        if self.executor:
            if self.is_global_executor:
                local_env = env
            return self.executor(args, self.body, local_env)
        for i, expr in enumerate(self.body):
            res = eval_parsed(expr, local_env)
            if res is None:
                continue
            if expr.type == "KEYWORD" and expr.value == "ret":
                if i + 1 < len(self.body):
                    return eval_parsed(self.body[i + 1], local_env)
            if isinstance(res, Instruction) and res.type == "RETURN_VALUE":
                return res.value
    
    def __repr__(self):
        return f"func {self.name}({', '.join(self.params_name)}) {{ ... }}"


class Node:
    def __init__(
        self,
        type,
        value=None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        children: Optional[list["Node"]] = None,
    ):
        self.type = type
        self.value = value
        self.left = left
        self.right = right
        self.children = children

    def __repr__(self):
        return f"{self.__class__.__name__}({self.type}, {self.value})"


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.length = len(tokens)
        self.pos = 0
    
    def skip_end(self):
        while self.token().type in ("ENDL", "ENDEXPR"):
            self.next()

    def statements(self):
        nodes: list[Node] = []
        while self.has_next_token():
            if self.token().type in ("ENDEXPR", "ENDL"):
                self.next()
                continue
            nodes.append(self.expression())
            if self.has_next_token() and self.token().type in ("ENDEXPR", "ENDL"):
                self.next()
        return nodes

    def token(self, add_pos: int = 0):
        return self.tokens[self.pos + add_pos]

    def check_type(self, token: Token, *excepted: str):
        return token.type in excepted if excepted else True

    def next(self, *excepted: str):
        if self.pos < self.length and self.check_type(self.tokens[self.pos], *excepted):
            self.pos += 1
            return
        raise SyntaxError(f"Next method error. Excepted {excepted}")

    def has_next_token(self):
        return self.pos < self.length - 1

    def expression(self) -> Node:
        node = self.and_handle()
        while self.has_next_token() and (token := self.token()).type in (
            "OR",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.and_handle())
        return node
    
    def and_handle(self) -> Node:
        node = self.cmp_op()
        while self.has_next_token() and (token := self.token()).type in (
            "AND",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.cmp_op())
        return node
    
    def cmp_op(self) -> Node:
        node = self.pm()
        while self.has_next_token() and (token := self.token()).type in (
            "LESS",
            "GREATER",
            "GREATER_EQ",
            "LESS_EQ",
            "EQUALS_CHECK",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.pm())
        return node
    
    def pm(self) -> Node:
        node = self.term()
        while self.has_next_token() and (token := self.token()).type in (
            "PLUS",
            "MINUS",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.term())
        return node

    def term(self) -> Node:
        node = self.factor()
        while self.has_next_token() and (token := self.token()).type in (
            "STAR",
            "SLASH",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.factor())
        return node

    def factor(self) -> Node:
        token = self.token()
        if not self.has_next_token():
            return Node(token.type, token.value)
        match token.type:
            case "PLUS" | "MINUS" | "STAR" | "SLASH":
                self.next()
                return Node(f"UNARY_{token.type}", right=self.factor())
            case "INCREMENT":
                self.next()
                return Node("PRE_INCREMENT", left=self.factor())
            case "DECREMENT":
                self.next()
                return Node("PRE_DECREMENT", left=self.factor())
            case "NUMBER" | "STRING_LITERAL":
                self.next()
                return Node(token.type, token.value)
            case "NAME":
                if not self.has_next_token():
                    return Node("NAME", token.value)
                match self.token(1).type:
                    case "LPAREN":
                        node = Node("CALL", token.value)
                        node = self.call_object(node)
                        self.next("RPAREN")
                        return node
                    case "EQUALS":
                        self.next()
                        self.next()
                        node = Node(
                            "ASSIGN",
                            left=Node(token.type, token.value),
                            right=self.expression(),
                        )
                        return node
                    case "INCREMENT":
                        self.next()
                        self.next()
                        return Node(
                            "POST_INCREMENT", left=Node(token.type, token.value)
                        )
                    case "DECREMENT":
                        self.next()
                        self.next()
                        return Node(
                            "POST_DECREMENT", left=Node(token.type, token.value)
                        )
                self.next()
                return Node("NAME", token.value)
            case "LPAREN":
                self.next()
                node = self.expression()
                self.next("RPAREN")
                return node
            case "KEYWORD":
                if not self.has_next_token():
                    raise SyntaxError
                match token.value:
                    case "delete":
                        if not (self.has_next_token() and self.token(1).type == "NAME"):
                            raise SyntaxError(f"Excepted identifier, got {self.token(1).value}".upper())
                        self.next()
                        val = self.token().value
                        self.next()
                        if self.has_next_token():
                            self.next()
                        return Node("DELETE", Node("NAME", val))
                    case "func":
                        return self.parse_function()
                    case "if":
                        node = self.parse_if()
                        children = [node]
                        self.skip_end()
                        while (token := self.token()).type == "KEYWORD" and self.token().value in ("elif", "else"):
                            if token.value == "elif":
                                parsed = self.parse_if()
                                self.skip_end()
                            else:
                                self.next()
                                parsed = self.parse_body()
                                self.skip_end()
                            children.append(Node(f"_{token.value.upper()}", children=[Node("BODY", children=parsed)]))
                        return Node("IF_STATEMENT", IF.from_list(children))
                    case "load":
                        self.next()
                        return Node(
                            "LOAD_MODULE", token.value, children=[self.factor()]
                        )
                    case "loop":
                        self.next()
                        if self.token().value not in ("while", "for"):
                            raise SyntaxError(f"Excepted loop spec, not {self.token().value}")
                        loop_type = self.token().value
                        self.next()
                        condition = self.factor()
                        body = self.parse_body()
                        return Node("LOOP", loop_type, children=[condition, Node("BODY", children=body)])
                self.next()
                return Node("KEYWORD", token.value)
            case "EQUALS":
                raise SyntaxError("Unexcepted token: '='")
            case "LBRACKET":
                self.next()
                return Node("LBRACKET", right=self.expression())
            case "RBRACKET":
                self.next()
                return Node("RBRACKET")
            case a if a not in ("ENDL", "ENDEXPR"):
                raise SyntaxError(f"Unexcepted token: '{token.value}'")

    def call_object(self, node: Node, anonymous: bool = False):
        def read_parameter():
            param = self.expression()
            return param

        def handle_parameters():
            while self.token().type == "COMMA":
                self.next()
                if self.token().type in ("COMMA", "RPAREN"):
                    continue
                node.children.append(read_parameter())

        node.children = []
        if not anonymous:
            self.next("NAME")
        self.next("LPAREN")
        if self.token().type != "RPAREN":
            node.children.append(read_parameter())
            handle_parameters()
        while self.has_next_token() and self.token(1).type == "LPAREN":
            self.next()
            self.next()
            node = Node("ANONYMOUS_CALL", left=node, children=[])
            if self.token().type != "RPAREN":
                node.children.append(read_parameter())
                handle_parameters()
        return node

    def parse_function(self) -> Node:
        self.next("KEYWORD")
        func_name = self.token()
        self.next("NAME")
        self.next("LPAREN")
        params = []
        while self.token().type != "RPAREN":
            if self.token().type == "NAME":
                params.append(self.token().value)
            self.next()
            if self.token().type == "COMMA":
                self.next()
        self.next("RPAREN")
        body = self.parse_body()
        return Node(
            "DEFINE_FUNCTION",
            func_name.value,
            children=[Node("PARAMETERS", params), Node("BODY", children=body)],
        )
    
    def parse_loop(self):
        ...

    def parse_if(self) -> Node:
        self.next("KEYWORD")
        condition = self.factor()
        body = self.parse_body()
        return Node(
            "_IF", children=[Node("PARAMETERS", condition), Node("BODY", children=body)]
        )
    
    def parse_else(self) -> Node:
        self.next("KEYWORD")
        body = self.parse_body()
        return Node(
            "_ELSE", children=[Node("BODY", children=body)]
        )
    
    def parse_body(self):
        self.skip_end() # !!
        self.next("LBRACKET")
        body = []
        while self.token().type != "RBRACKET":
            body.append(self.expression())
            if self.token().type in ("ENDEXPR", "ENDL"):
                self.next()
        self.next("RBRACKET")
        return body
    

def exec_body(body_node: Node, env):
    nodes = list(filter(None, body_node.children))
    for index, i in enumerate(nodes):
        ret = eval_parsed(i, env)
        if isinstance(ret, Instruction):
            if ret.type == "RETURN" and index + 1 < len(nodes):
                return Instruction(
                    "RETURN_VALUE", eval_parsed(nodes[index + 1], env)
                )
            return ret
        

def exec_loop_body(condition_node: Node, body_node: Node, env):
    nodes = list(filter(None, body_node.children))
    running = True
    while running and eval_parsed(condition_node, env):
        for index, i in enumerate(nodes):
            i = nodes[index]
            ret = eval_parsed(i, env)
            if isinstance(ret, Instruction):
                if ret.type == "RETURN" and index + 1 < len(nodes):
                    return Instruction(
                        "RETURN_VALUE", eval_parsed(nodes[index + 1], env)
                    )
                if ret.type == "RETURN_VALUE":
                    return ret
                elif ret.type == "BREAK":
                    running = False
                    break
                elif ret.type == "CONTINUE":
                    break

def eval_parsed(node: Node, env: Environment):
    if not node:
        return
    match node.type:
        case "AND":
            return int(bool(eval_parsed(node.left, env) and eval_parsed(node.right, env)))
        case "OR":
            return int(bool(eval_parsed(node.left, env) or eval_parsed(node.right, env)))
        case "GREATER":
            return int(bool(eval_parsed(node.left, env) > eval_parsed(node.right, env)))
        case "GREATER_EQ":
            return int(bool(eval_parsed(node.left, env) >= eval_parsed(node.right, env)))
        case "LESS":
            return int(bool(eval_parsed(node.left, env) < eval_parsed(node.right, env)))
        case "LESS_EQ":
            return int(bool(eval_parsed(node.left, env) <= eval_parsed(node.right, env)))
        case "EQUALS_CHECK":
            return int(bool(eval_parsed(node.left, env) == eval_parsed(node.right, env)))
        case "PRE_INCREMENT":
            return eval_parsed(node.left, env) + 1
        case "POST_INCREMENT":
            return eval_parsed(node.left, env) + 1
        case "PRE_DECREMENT":
            return eval_parsed(node.left, env) - 1
        case "POST_DECREMENT":
            return eval_parsed(node.left, env) - 1
        case "NAME":
            return val if (val := env.get(node.value)) is not None else "NULL"
        case "NUMBER":
            return int(node.value)
        case "STRING_LITERAL":
            return str(node.value)[1:-1]
        case "PLUS":
            return eval_parsed(node.left, env) + eval_parsed(node.right, env)
        case "UNARY_PLUS":
            return +eval_parsed(node.right, env)
        case "MINUS":
            return eval_parsed(node.left, env) - eval_parsed(node.right, env)
        case "UNARY_MINUS":
            return -eval_parsed(node.right, env)
        case "SLASH":
            return eval_parsed(node.left, env) // eval_parsed(node.right, env)
        case "STAR":
            return eval_parsed(node.left, env) * eval_parsed(node.right, env)
        case "UNARY_STAR":
            return eval_parsed(node.right, env) * 1
        case "CALL":
            parameters = node.children.copy()
            for i, val in enumerate(parameters):
                parameters[i] = eval_parsed(val, env)
            func: Function = env.get(node.value)
            return func.exec_body(parameters)
        case "ANONYMOUS_CALL":
            parameters = node.children.copy()
            for i, val in enumerate(parameters):
                parameters[i] = eval_parsed(val, env)
            if node.left:
                node.left = eval_parsed(node.left, env)
            if type(node.left) == Function:
                return node.left.exec_body(parameters)
            raise TypeError("cannot call anonymous object that is not callable")
        case "ASSIGN":
            left, right = node.left.value, eval_parsed(node.right, env)
            if (i := isinstance(right, Instruction)) and node.right.type != "IF_STATEMENT":
                raise SyntaxError("Invalid right operand for '='")
            env.set(left, right if not i else right.value)
            return env.get(left)
        case "DEFINE_FUNCTION":
            func_name = node.value
            params = node.children[0].value
            body = node.children[1].children
            env.set(func_name, Function(len(params), body, params, env, name=func_name))
        case "LOAD_MODULE":
            if len(node.children) != 1 or node.children[0].type != "STRING_LITERAL":
                raise SyntaxError
            with open(f"{eval_parsed(node.children[0], env)}.s") as module:
                statements = Parser(get_tokens(module.read())).statements()
                for node in statements:
                    eval_parsed(node, env)

        case "IF_STATEMENT":
            node: IF = node.value
            If, Elifs, Else = node.If, node.Elifs, node.Else
            if eval_parsed(If.children[0].value, env):
                return exec_body(If.children[-1], env)
            for If in Elifs:
                If: Node = If.children[0].children
                if eval_parsed(If.children[0].value, env):
                    return exec_body(If.children[-1], env)
            return eval_parsed(Else.children[0], env) if Else else "NULL"
        case "LOOP":
            condition, body = node.children
            return exec_loop_body(condition, body, env)

        case "BODY":
            return exec_body(node, env)
        case "KEYWORD":
            if node.value == "ret":
                return Instruction("RETURN")
            elif node.value in ("break", "continue"):
                return Instruction(node.value.upper())
            
        case "DELETE":
            if node.value.value in env.vars:
                env.vars.pop(node.value.value)

def run_module(modulename: str):
    with open(f"{modulename}.s") as f:
        code = f.read()

    print_err = lambda txt, err_type="RUNTIME ERROR": print(
        f"\033[31m[{err_type}]:\033[33m {txt}\033[0m"
    )
    try:
        parser = Parser(t := get_tokens(code))
        expr = parser.statements()
    except SyntaxError as e:
        print_err(e, "TOKENIZATION ERROR")
        exit()
    for i in expr:
        try:
            eval_parsed(i, env)
        except Exception as e:
            et = type(e)
            if et == FileNotFoundError:
                print_err("Cannot find module".upper())
            elif et == SyntaxError:
                print_err(e)
            elif et == TypeError:
                print_err(f"TYPE ERROR: {e}".upper())
            elif et == AttributeError:
                print_err(f"CANNOT CALL THIS OBJECT".upper())
            else:
                print_err(str(e).upper())
        except KeyboardInterrupt as e:
            print_err("KEYBOARD INTERRUPT")


env = Environment()
env.set("EXECUTOR_BODY", "lambda args, body, env")
env.set("writeln", Function(-1, executor=lambda args, body, env: print(*args)))
env.set("readln", Function(1, executor=lambda args, body, env: input(args[0])))
env.set(
    "length",
    Function(
        1,
        executor=lambda args, body, env: len(args[0])
    ),
)
env.set("envp", Function(-1, executor=lambda args, body, env: print(env.vars)))
env.set("SYSEXEC", Function(1, executor=lambda args, body, env: eval(args[0])))
env.set(
    "REGISTER",
    Function(
        3,
        executor=lambda args, body, env: env.set(
            args[0], Function(args[1], executor=args[2], name=args[0])
        ),
        global_executor_env=True,
    ),
)
env.set(
    "CREATE_BASIC_EXECUTOR",
    Function(
        1,
        executor=lambda a, b, e: lambda args, body, env: eval(a[0]),
        global_executor_env=True,
    ),
)


if __name__ == "__main__":
    run_module("main")