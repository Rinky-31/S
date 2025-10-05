class Base:
    def __init__(self):
        self.name = getattr(self, "name", self.__class__.__name__)
        self.attrs = {
            "toString": self.__repr__,
            "_call": getattr(self, "_call", None),
        }
    
    def get_attr(self, name: str):
        return self.attrs.get(name)

    def __repr__(self):
        return f"instance of {self.name or '...'} {{ ... }}"


class Integer(Base):
    def __init__(self, value):
        super().__init__()
        self.value = int(value)

    def __increment(self, is_post: bool):
        res = Integer(self.value) if is_post else self
        self.value += 1
        return res
    
    def __decrement(self, is_post: bool):
        res = Integer(self.value) if is_post else self
        self.value += 1
        return res
    
    def _negative(self):
        self.value = -self.value 
        return self
    
    def _positive(self):
        return self
    
    def _call(self):
        raise

    def get_attr(self, name: str):
        match name:
            case "_increment":
                return self.__increment
            case "_get_item":
                return lambda i: print(i)

    def _greater(self, o: "Integer"):
        return self.value > o.value