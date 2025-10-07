from lexer import Token, get_tokens
from typing import Optional
from sys import argv
from base_classes import *


log_mode = False

def log(text: str):
    if log_mode:
        print(f"\033[33m[LOG]: \033[32m{text}\033[0m")

class Instruction:
    def __init__(self, type: str, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.type}, {self.value})"


class IF:
    def __init__(
        self,
        ifs: Optional["Node"] = None,
        Elifs: Optional[list["Node"]] = None,
        Else: Optional["Node"] = None,
    ):
        self.If = ifs
        self.Elifs = Elifs or []
        self.Else = Else
        log(f"Created if ({len(self.Elifs)} elifs)")

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


class Switch:
    def __init__(
        self,
        cases: Optional[list["Node"]] = None,
        another: Optional["Node"] = None,
        value=None,
    ):
        self.cases = cases or []
        self.another = another
        self.value = value
        log(f"Created switch ({len(self.cases)} cases)")

    @staticmethod
    def from_list(lst: list["Node"]):
        cases, another = [], None
        for i in lst:
            if i.type == "_ANOTHER":
                another = i
            elif i.type == "_CASE":
                cases.append(i)
        return Switch(cases, another)


class Null:
    __i = None

    def __new__(cls):
        if cls.__i is None:
            cls.__i = super().__new__(cls)
        return cls.__i

    def __repr__(self):
        return self.__class__.__name__

    def __bool__(self):
        return False


class Function(Base):
    def __init__(
        self,
        body: list["Node"] = [],
        params: Optional[list["Node"]] = None,
        _env: Optional[Environment] = None,
        *,
        executor=None,
        name: Optional[str] = None,
        global_executor_env: bool = False,
        executor_min_argcount: int = 0,
    ):
        super().__init__()
        self.body = list(filter(lambda a: a is not None, body))
        self.executor = executor
        self.env = Environment(parent=_env or env)
        self.is_global_executor = global_executor_env
        self.params = params or []
        self.name = name
        self.executor_argcount = executor_min_argcount
        log(f"Created function {self}")

    def _call(self, args: list, env: Optional[Environment] = None):
        log(f"Calling function {self}")
        args = list(filter(lambda a: a is not None, args)) if args else []
        local_env = Environment(parent=(env or self.env))
        params_name = [param.value for param in self.params]
        local_env.vars.update(parameters := tuple(zip(params_name, args)))
        for param in self.params:
            if param.type == "REQUIRED_PARAMETER" and param.value:
                if not any(p == param.value for p, _ in parameters):
                    raise TypeError(
                        f"Function missing required parameter '{param.value}'"
                    )
        if self.name:
            local_env.set(self.name, self)
        if self.executor:
            if self.is_global_executor:
                local_env = env
            if len(args) < self.executor_argcount:
                raise TypeError(
                    f"Not enought arguments. Excepted {self.executor_argcount}, got {len(parameters)}"
                )
            log(f"Calling function {self.name or '...'} executor")
            return self.executor(args, self.body, local_env)
        for i, expr in enumerate(self.body):
            res = eval_parsed(expr, local_env)
            if res is None:
                continue
            if expr.type == "KEYWORD" and expr.value == "ret":
                if i + 1 < len(self.body):
                    res = eval_parsed(self.body[i + 1], local_env)
                    log(f"Return {res} from {self}")
                    return res
            if isinstance(res, Instruction) and res.type == "RETURN_VALUE":
                log(f"Return {res.value} from {self}")
                return res.value
        log(f"Return Null from {self}")
        return Null()

    def __repr__(self):
        return f"func {self.name or '...'}({', '.join(param.value for param in self.params)}) {{ ... }}"


class ClassPrototype(Base):
    def __init__(
        self,
        body: list["Node"],
        _env: Environment,
        *,
        name: Optional[str] = None,
    ):
        log(f"Creating class prototype")
        super().__init__()
        self.body = list(filter(lambda a: a is not None, body))
        self.env = Environment(parent=_env)
        self.name = name
        exec_body(Node("BODY", children=self.body), self.env)
        log(f"Created {self}")

    def _call(self, params):
        instance = ClassInstance(
            self.env,
            name=self.name,
            params=params,
            constructor=self.env.vars.get("_ctor"),
            destructor=self.env.vars.get("_dtor"),
        )
        return instance

    def get_attr(self, name: str):
        if name in ("_call", "_get_item"):
            return super().get_attr(name)
        return self.env.vars.get(name) or Null()

    def set_attr(self, name: str, value):
        self.env.set(name, value)

    def __repr__(self):
        return f"class prototype {self.name or '...'} {{ ... }}"


class ClassInstance(Base):
    def __init__(
        self,
        _env: Optional[Environment] = None,
        *,
        name: Optional[str] = None,
        params: Optional[list] = None,
        constructor: Optional[Function] = None,
        destructor: Optional[Function] = None,
    ):
        log(f"Creating class instance")
        super().__init__()
        self.env = Environment(parent=_env or env)
        self.name = name or self.name
        self.params = params
        self.env.set("this", self)
        if constructor and isinstance(constructor, Function):
            constructor._call(params, self.env)
        self.destructor = destructor
        log(f"Created instance of {self.name}")

    def get_attr(self, name: str):
        res = self.env.vars.get(name) or self.env.parent.vars.get(name) or Null()
        if not res:
            if (g := self.env.parent.vars.get("_get_attr")):
                return getattr(g, "_call", g)([name])
            return res
        if isinstance(res, (Function)):
            return lambda args = None: res._call(args, self.env)
        return res

    def set_attr(self, name: str, value):
        self.env.set(name, value)

    def delete(self):
        if isinstance(self.destructor, Function):
            self.destructor._call([], self.env)

    def __repr__(self):
        return f"instance of {self.name or '...'} {{ ... }}"


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
        while self.has_next_token() and self.token().type in ("ENDL", "ENDEXPR"):
            self.next()

    def statements(self):
        nodes: list[Node] = []
        while self.has_next_token():
            if self.token().type in ("ENDEXPR", "ENDL"):
                self.next()
                continue
            nodes.append(self.assign_handle())
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

    def assign_handle(self) -> Node:
        node = self.expression()
        if self.has_next_token() and (token := self.token()).type in ("EQUALS",):
            self.next()
            node = Node(token.type, left=node, right=self.assign_handle())
        return node

    def expression(self) -> Node:
        node = self.and_handle()
        while self.has_next_token() and (token := self.token()).type in ("OR",):
            self.next()
            node = Node(token.type, left=node, right=self.and_handle())
        return node

    def and_handle(self) -> Node:
        node = self.handle_contains()
        while self.has_next_token() and (token := self.token()).type in ("AND",):
            self.next()
            node = Node(token.type, left=node, right=self.handle_contains())
        return node

    def handle_contains(self) -> Node:
        node = self.cmp_op()
        while self.has_next_token() and (token := self.token()).type in ("CONTAINS",):
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
            "NOT_EQUALS_CHECK",
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
        node = self.handle_attributes()
        while self.has_next_token() and (token := self.token()).type in (
            "STAR",
            "SLASH",
        ):
            self.next()
            node = Node(token.type, left=node, right=self.handle_attributes())
        return node

    def handle_attributes(self) -> Node:
        node = self.after_factor()
        if self.has_next_token() and (token := self.token()).type in ("OF",):
            self.next()
            node = Node(token.type, left=node, right=self.handle_attributes())
        return node

    def after_factor(self) -> Node:
        node = self.factor()
        while self.has_next_token() and (token := self.token()).type in ("LPAREN", "LSQBRACKET"):
            if node.type == "KEYWORD":
                return node
            node = Node("CALL" if token.type == "LPAREN" else "GET_ITEM", left=node)
            node = (self.call_object if token.type == "LPAREN" else self.get_item)(node)
            self.next("RPAREN" if token.type == "LPAREN" else "RSQBRACKET")
        return node

    def factor(self) -> Node:
        token = self.token()
        if not self.has_next_token():
            return Node(token.type, token.value)
        match token.type:
            case "COMMA":
                raise SyntaxError("Unexcepted token: ','")
            case "PLUS" | "MINUS" | "STAR" | "SLASH":
                self.next()
                return Node(f"UNARY_{token.type}", token.value, right=self.factor())
            case "INCREMENT":
                self.next()
                return Node("PRE_INCREMENT", left=self.factor())
            case "DECREMENT":
                self.next()
                return Node("PRE_DECREMENT", left=self.factor())
            case "NUMBER" | "STRING_LITERAL" | "FLOAT_NUMBER":
                self.next()
                return Node(token.type, token.value)
            case "NAME":
                if not self.has_next_token():
                    return Node("NAME", token.value)
                match self.token(1).type:
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
                node = self.assign_handle()
                self.next("RPAREN")
                return node
            case "LBRACKET" | "RBRACKET":
                self.next()
                return Node(token.type, token.value)
            case "KEYWORD":
                if not self.has_next_token():
                    raise SyntaxError
                match token.value:
                    case "delete":
                        if not (self.has_next_token() and self.token(1).type == "NAME"):
                            raise SyntaxError(
                                f"Excepted identifier, got {self.token(1).value}".upper()
                            )
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
                        while (
                            self.has_next_token()
                            and (token := self.token()).type == "KEYWORD"
                            and self.token().value in ("elif", "else")
                        ):
                            if token.value == "elif":
                                parsed = self.parse_if()
                                self.skip_end()
                            else:
                                self.next()
                                parsed = self.parse_body()
                                self.skip_end()
                            children.append(
                                Node(
                                    f"_{token.value.upper()}",
                                    children=[Node("BODY", children=parsed)],
                                )
                            )
                        return Node("IF_STATEMENT", IF.from_list(children))
                    case "switch":
                        self.next("KEYWORD")
                        value_to_check = self.factor()
                        self.next("LBRACKET")
                        self.skip_end()
                        children = []
                        while (
                            self.has_next_token()
                            and (token := self.token()).type == "KEYWORD"
                            and self.token().value in ("case", "another")
                        ):
                            if token.value == "case":
                                parsed = self.parse_case()
                                self.skip_end()
                            else:
                                self.next()
                                parsed = Node(
                                    "_ANOTHER",
                                    children=[Node("BODY", children=self.parse_body())],
                                )
                                self.skip_end()
                            children.append(parsed)
                        res = Switch.from_list(children)
                        res.value = value_to_check
                        self.next("RBRACKET")
                        return Node("SWITCH_STATEMENT", res)

                    case "load":
                        self.next()
                        return Node(
                            "LOAD_MODULE", token.value, children=[self.factor()]
                        )
                    case "loop":
                        self.next()
                        if self.token().value not in ("while", "for"):
                            raise SyntaxError(
                                f"Excepted loop spec, not {self.token().value}"
                            )
                        loop_type = self.token().value
                        self.next()
                        condition = self.factor()
                        body = self.parse_body()
                        return Node(
                            "LOOP",
                            loop_type,
                            children=[condition, Node("BODY", children=body)],
                        )
                    case "class":
                        return self.parse_class()
                    case "drop":
                        self.next()
                        return Node("DROP", self.assign_handle())

                self.next()
                return Node("KEYWORD", token.value)

    def call_object(self, node: Node):
        def read_parameter():
            param = self.assign_handle()
            return param

        def handle_parameters():
            while self.token().type == "COMMA":
                self.next()
                if self.token().type in ("COMMA", "RPAREN"):
                    continue
                node.children.append(read_parameter())

        node.children = []
        self.next("LPAREN")
        if self.token().type != "RPAREN":
            node.children.append(read_parameter())
            handle_parameters()
        return node
    
    def get_item(self, node: Node):
        def read_parameter():
            param = self.assign_handle()
            return param

        def handle_parameters():
            while self.token().type == "COMMA":
                self.next()
                if self.token().type in ("COMMA", "RPAREN"):
                    continue
                node.children.append(read_parameter())

        node.children = []
        self.next("LSQBRACKET")
        if self.token().type != "RSQBRACKET":
            node.children.append(read_parameter())
            handle_parameters()
        return node

    def parse_function(self) -> Node:
        self.next("KEYWORD")
        func_name = self.token()
        self.next("NAME", "LPAREN")
        log(f"Declaration function {func_name.value if func_name.type != 'LPAREN' else '...'}")
        if self.token().type == "LPAREN":
            self.next("LPAREN")
        params = []
        while self.token().type != "RPAREN":
            token = self.token()
            if token.type == "KEYWORD" and token.value == "req":
                if self.token(1).type != "NAME":
                    raise SyntaxError("Excepted identifier")
                self.next()
                params.append(Node("REQUIRED_PARAMETER", self.token().value))
            if token.type == "NAME":
                params.append(Node("UNREQUIRED_PARAMETER", token.value))
            self.next()
            if self.token().type == "COMMA":
                self.next()
        self.next("RPAREN")
        body = self.parse_body()
        return Node(
            "DEFINE_FUNCTION",
            func_name.value if func_name.type == "NAME" else None,
            children=[Node("PARAMETERS", params), Node("BODY", children=body)],
        )

    def parse_class(self):
        self.next("KEYWORD")
        class_name = self.token()
        if class_name.type == "NAME":
            self.next()
        body = self.parse_body()
        return Node(
            "DEFINE_CLASS",
            class_name.value if class_name.type == "NAME" else None,
            children=[Node("BODY", children=body)],
        )

    def parse_if(self) -> Node:
        self.next("KEYWORD")
        condition = self.factor()
        body = self.parse_body()
        return Node(
            "_IF", children=[Node("PARAMETERS", condition), Node("BODY", children=body)]
        )

    def parse_case(self) -> Node:
        self.next("KEYWORD")
        condition = self.factor()
        body = self.parse_body()
        return Node(
            "_CASE",
            children=[Node("PARAMETERS", condition), Node("BODY", children=body)],
        )

    def parse_body(self):
        self.skip_end()
        self.next("LBRACKET")
        body = []
        while self.token().type != "RBRACKET":
            body.append(self.assign_handle())
            if self.token().type in ("ENDEXPR", "ENDL"):
                self.next()
        self.next("RBRACKET")
        return body


def exec_body(body_node: Node, env):
    nodes = list(filter(lambda a: a is not None, body_node.children))
    for index, i in enumerate(nodes):
        ret = eval_parsed(i, env)
        if isinstance(ret, Instruction):
            if ret.type == "RETURN" and index + 1 < len(nodes):
                return Instruction("RETURN_VALUE", eval_parsed(nodes[index + 1], env))
            return ret


def exec_loop_body(condition_node: Node, body_node: Node, env):
    while eval_parsed(condition_node, env):
        ret = exec_body(body_node, env)
        if isinstance(ret, Instruction):
            if ret.type == "RETURN_VALUE":
                return ret
            elif ret.type == "BREAK":
                break
            elif ret.type == "CONTINUE":
                continue


def eval_parsed(node: Node, env: Environment):
    if not node:
        return
    match node.type:
        case "DROP":
            raise Exception(eval_parsed(node.value, env))
        case "OF":
            attribute, from_object = node.left, eval_parsed(node.right, env)
            match attribute.type:
                case "CALL" | "GET_ITEM":
                    left, prev = attribute.left, attribute
                    while left.type in ("CALL", "GET_ITEM"):
                        prev = left
                        left = left.left
                    if left.type != "NAME":
                        raise SyntaxError("Excepted identifier")
                    attr = (
                        from_object.get_attr(left.value)
                        if hasattr(from_object, "get_attr")
                        else getattr(from_object, left.value)
                    )
                    if not attr:
                        raise TypeError("Object have not this attribute")
                    prev.left = Node("ATTRIBUTE", attr, left)
                    return eval_parsed(
                        attribute,
                        (
                            env
                            if not isinstance(
                                from_object, (ClassPrototype, ClassInstance)
                            )
                            else from_object.env
                        ),
                    )
                case "NAME":
                    if isinstance(from_object, Environment):
                        return from_object.get(attribute.value)
                    if attribute.value == "length":
                        return from_object.__len__()
                    return (
                        from_object.get_attr(attribute.value)
                        if hasattr(from_object, "get_attr")
                        else getattr(from_object, attribute.value)
                    )
                case _:
                    raise SyntaxError("Excepted identifier")
        case "AND":
            return int(
                bool(eval_parsed(node.left, env) and eval_parsed(node.right, env))
            )
        case "OR":
            return int(
                bool(eval_parsed(node.left, env) or eval_parsed(node.right, env))
            )
        case "CONTAINS":
            return int(
                bool(eval_parsed(node.right, env) in eval_parsed(node.left, env))
            )
        case "GREATER":
            return int(bool(eval_parsed(node.left, env) > eval_parsed(node.right, env)))
        case "GREATER_EQ":
            return int(
                bool(eval_parsed(node.left, env) >= eval_parsed(node.right, env))
            )
        case "LESS":
            return int(bool(eval_parsed(node.left, env) < eval_parsed(node.right, env)))
        case "LESS_EQ":
            return int(
                bool(eval_parsed(node.left, env) <= eval_parsed(node.right, env))
            )
        case "EQUALS_CHECK":
            return int(
                bool(eval_parsed(node.left, env) == eval_parsed(node.right, env))
            )
        case "NOT_EQUALS_CHECK":
            return int(
                bool(eval_parsed(node.left, env) != eval_parsed(node.right, env))
            )
        case "PRE_INCREMENT":
            return eval_parsed(node.left, env).get_attr("_increment")([0])
        case "POST_INCREMENT":
            return eval_parsed(node.left, env).get_attr("_increment")([1])
        case "PRE_DECREMENT":
            return eval_parsed(node.left, env) - 1
        case "POST_DECREMENT":
            # ...
            return eval_parsed(node.left, env) - 1
        case "NAME":
            return val if (val := env.get(node.value)) is not None else Null()
        case "NUMBER":
            return int(node.value)
        case "FLOAT_NUMBER":
            return float(node.value)
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
        case "UNARY_STAR" | "UNARY_SLASH":
            raise SyntaxError(f"Unsupported unary operation in this version: {node.value}")
        case "ATTRIBUTE":
            name = node.left.value
            value = node.value
            node.type = "NAME"
            node.value = name
            return value
        case "CALL" | "GET_ITEM":
            parameters = [eval_parsed(val, env) for val in node.children]
            if node.left:
                func = eval_parsed(node.left, env)
            if node.type == "CALL":
                if hasattr(func, "get_attr") and (call := func.get_attr("_call")):
                    return getattr(call, "_call", call)(parameters)
                return func(*parameters)
            elif node.type == "GET_ITEM":
                if hasattr(func, "get_attr") and (call := func.get_attr("_get_item")):
                    return getattr(call, "_call", call)(parameters)
                return func[*parameters]
        case "EQUALS":
            if (
                node.left.type not in ("NAME", "OF")
                or node.left.type == "NAME"
                and node.left.value in reserved_names
            ):
                raise SyntaxError(f"Invalid left operand for '=': {node.left.value or node.left.type}")
            if node.left.type == "OF":
                attr_name = node.left.left.value
                obj = eval_parsed(node.left.right, env)
                if hasattr(obj, "set_attr"):
                    obj.set_attr(attr_name, res := eval_parsed(node.right, env))
                    return res
                setattr(obj, attr_name, eval_parsed(node.right, env))
                return
            if node.right.type == "LOAD_MODULE":
                if node.right.children[0].value.strip('"') in loaded_modules:
                    return
                eval_parsed(node.right, env)
                _env = Environment(env.vars.copy())
                env.set(node.left.value, _env)
                return _env
            left, right = node.left.value, eval_parsed(node.right, env)
            if (i := isinstance(right, Instruction)) and node.right.type not in (
                "IF_STATEMENT",
                "SWITCH_STATEMENT",
            ):
                raise SyntaxError("Invalid right operand for '='")
            env.set(left, val := right if not i else right.value)
            return val

        case "DEFINE_FUNCTION":
            func_name = node.value
            params = node.children[0].value
            body = node.children[1].children
            f = Function(body, params, env, name=func_name)
            if func_name:
                env.set(func_name, f)
            return f

        case "DEFINE_CLASS":
            body = node.children[0].children
            cls = ClassPrototype(body, env, name=node.value)
            if node.value:
                env.set(node.value, cls)
            return cls

        case "LOAD_MODULE":
            if len(node.children) != 1 or node.children[0].type != "STRING_LITERAL":
                raise SyntaxError
            name = node.children[0].value.strip('"')
            if name in loaded_modules:
                log(f"Already loaded: {node.children[0].value}")
                for key, val in loaded_modules[name].vars.items():
                    env.set(key, val)
                return
            log(f"Loading module {name}")
            with open(
                f"{eval_parsed(node.children[0], env)}.epy", encoding="utf-8"
            ) as module:
                statements = Parser(get_tokens(module.read())).statements()
                log(f"Got {len(statements)} nodes for exex")
                loaded_modules[name] = (_e := Environment(parent=env))
                for node in statements:
                    eval_parsed(node, _e)
                for key, val in _e.vars.items():
                    env.set(key, val)
            log(f"Loaded module {name}")

        case "IF_STATEMENT":
            node: IF = node.value
            If, Elifs, Else = node.If, node.Elifs, node.Else
            if eval_parsed(If.children[0].value, env):
                return exec_body(If.children[-1], env)
            for If in Elifs:
                If: Node = If.children[0].children
                if eval_parsed(If.children[0].value, env):
                    return (
                        res
                        if (res := exec_body(If.children[-1], env)) is not None
                        else Instruction("RETURN_VALUE", Null())
                    )
            return (
                eval_parsed(Else.children[0], env)
                if Else
                else Null()
            )
        case "SWITCH_STATEMENT":
            switch: Switch = node.value
            value_to_check = eval_parsed(switch.value, env)
            another = switch.another
            for case in switch.cases:
                if eval_parsed(case.children[0].value, env) == value_to_check:
                    return (
                        res
                        if (res := exec_body(case.children[-1], env)) is not None
                        else Null()
                    )
            return (
                eval_parsed(another.children[0], env)
                if another
                else Null()
            )

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
                log(f"Deleting {node.value.value}")
                if hasattr(obj := env.vars[node.value.value], "delete"):
                    obj.delete()
                env.vars.pop(node.value.value)


def run_module(modulename: str):
    with open(f"{modulename}.epy", encoding="utf-8") as f:
        code = f.read()
    log("Program started")
    print_err = lambda txt, err_type="RUNTIME ERROR": print(
        f"\033[31m[{err_type}]:\033[33m {txt}\033[0m"
    )
    try:
        parser = Parser(t := get_tokens(code))
        expr = parser.statements()
        log(f"Got {len(expr)} nodes for exec")
    except SyntaxError as e:
        print_err(e, "TOKENIZATION ERROR")
        exit()
    for i in expr:
        try:
            eval_parsed(i, env)
        except Exception as e:
            et = type(e)
            if et == FileNotFoundError:
                print_err(f"PATH ERROR: {e}")
            elif et == SyntaxError:
                print_err(e)
            elif et == TypeError:
                print_err(f"TYPE ERROR: {e}")
            elif et == AttributeError:
                print_err(f"ATTR_ERROR: {e}")
            else:
                print_err(str(e))
            break
        except KeyboardInterrupt as e:
            print_err("KEYBOARD INTERRUPT")
            break
    log("Program end")


env = Environment()
env.set(
    "writeln",
    Function(
        executor=lambda args, body, env: print(
            *(
                (
                    (attr._call([]) if hasattr(attr, "_call") else attr())
                    if hasattr(a, "get_attr")
                    and not isinstance(a, ClassPrototype)
                    and (attr := a.get_attr("toString"))
                    else a
                )
                for a in args
            )
        ),
        name="writeln"
    ),
)
env.set(
    "readln",
    Function(executor=lambda args, body, env: input(args[0] if args else ">> ")),
)
env.set(
    "SYSEXEC",
    Function(executor=lambda args, body, env: eval(args[0]), executor_min_argcount=1),
)
env.set("Null", Null())
env.set("true", 1)
env.set("false", 0)

reserved_names: list[str] = ["Null", "false", "true", "this"]
loaded_modules: dict[str, Environment] = {}

if __name__ == "__main__":
    _, *flags = argv
    if flags:
        log_mode = flags[0] == "--log"
    run_module("main")
