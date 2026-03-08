class TreeNode:
    def __init__(self, label, children=None, lexeme=None):
        self.label = label
        self.children = children if children is not None else []
        self.lexeme = lexeme

    def __str__(self):
        return self._display()

    def _display(self, level=0):
        indent = "  " * level
        if self.lexeme:
            result = f"{indent}{self.label} ({self.lexeme.value})\n"
        else:
            result = f"{indent}{self.label}\n"
            for child in self.children:
                result += child._display(level + 1)
        return result