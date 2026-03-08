from lexical_analyzer import LexemeType, Lexeme
from syntax_tree import TreeNode

class QueryParser:
    def __init__(self, lexemes):
        self.lexemes = lexemes
        self.position = 0

    def current(self):
        return self.lexemes[self.position] if self.position < len(self.lexemes) else None

    def consume(self, *types):
        lex = self.current()
        if lex and lex.type in types:
            self.position += 1
            return lex
        return None

    def parse_root(self):
        start = self.position
        action_lex = self.consume(LexemeType.ACTION)
        if not action_lex:
            return None
        action_node = TreeNode("ACTION", lexeme=action_lex)

        entities_node = self.parse_entities()
        if not entities_node:
            self.position = start
            return None

        filters_node = self.parse_filters()
        if self.current() is not None:
            self.position = start
            return None

        children = [action_node, entities_node]
        if filters_node:
            children.append(filters_node)
        return TreeNode("ROOT", children=children)

    def parse_entities(self):
        start = self.position
        children = []

        quant = self.consume(LexemeType.QUANTIFIER)
        if quant:
            children.append(TreeNode("QUANTIFIER", lexeme=quant))

        entity = self.consume(LexemeType.ENTITY)
        if not entity:
            self.position = start
            return None
        children.append(TreeNode("ENTITY", lexeme=entity))

        while True:
            conn = self.consume(LexemeType.CONNECTOR)
            if not conn:
                break
            children.append(TreeNode("CONNECTOR", lexeme=conn))

            quant2 = self.consume(LexemeType.QUANTIFIER)
            if quant2:
                children.append(TreeNode("QUANTIFIER", lexeme=quant2))

            entity2 = self.consume(LexemeType.ENTITY)
            if not entity2:
                self.position = start
                return None
            children.append(TreeNode("ENTITY", lexeme=entity2))

        return TreeNode("ENTITIES", children=children)

    def parse_filters(self):
        children = []
        while True:
            filt = self.parse_filter()
            if filt:
                children.append(filt)
            else:
                break
        if not children:
            return None
        return TreeNode("FILTERS", children=children)

    def parse_filter(self):
        start = self.position

        person_filter = self.parse_person_filter()
        if person_filter:
            return TreeNode("FILTER", children=[person_filter])

        self.position = start
        topic_filter = self.parse_topic_filter()
        if topic_filter:
            return TreeNode("FILTER", children=[topic_filter])

        self.position = start
        year_filter = self.parse_year_filter()
        if year_filter:
            return TreeNode("FILTER", children=[year_filter])

        self.position = start
        return None

    def parse_person_filter(self):
        start = self.position
        children = []

        person = self.consume(LexemeType.PERSON)
        if not person:
            return None
        children.append(TreeNode("PERSON", lexeme=person))

        while True:
            # пропускаем запятые
            while self.current() and self.current().type == LexemeType.COMMA:
                self.consume(LexemeType.COMMA)

            conn = self.consume(LexemeType.CONNECTOR)
            if not conn:
                break
            children.append(TreeNode("CONNECTOR", lexeme=conn))

            person2 = self.consume(LexemeType.PERSON)
            if not person2:
                self.position = start
                return None
            children.append(TreeNode("PERSON", lexeme=person2))

        return TreeNode("PERSON_FILTER", children=children)

    def parse_topic_filter(self):
        start = self.position
        children = []

        prep = self.consume(LexemeType.PREP)
        if not prep:
            return None
        children.append(TreeNode("PREP", lexeme=prep))

        # собираем все слова темы
        topic_words = []
        while True:
            lex = self.current()
            if not lex:
                break
            if lex.type in [LexemeType.PREP, LexemeType.CONNECTOR, LexemeType.YEAR_NUM,
                            LexemeType.POST, LexemeType.PRIOR, LexemeType.INSIDE, LexemeType.PUBLISH_MARK]:
                break
            self.position += 1
            topic_words.append(TreeNode("WORD", lexeme=lex))
        if not topic_words:
            self.position = start
            return None
        children.extend(topic_words)

        # обработка союзов и дополнительных тем
        while True:
            conn = self.consume(LexemeType.CONNECTOR)
            if not conn:
                break
            children.append(TreeNode("CONNECTOR", lexeme=conn))

            prep2 = self.consume(LexemeType.PREP)
            if prep2:
                children.append(TreeNode("PREP", lexeme=prep2))

            more_words = []
            while True:
                lex = self.current()
                if not lex or lex.type in [LexemeType.PREP, LexemeType.CONNECTOR, LexemeType.YEAR_NUM,
                                           LexemeType.POST, LexemeType.PRIOR, LexemeType.INSIDE, LexemeType.PUBLISH_MARK]:
                    break
                self.position += 1
                more_words.append(TreeNode("WORD", lexeme=lex))
            if not more_words:
                self.position = start
                return None
            children.extend(more_words)

        return TreeNode("TOPIC_FILTER", children=children)

    def parse_year_filter(self):
        start = self.position

        published = self.consume(LexemeType.PUBLISH_MARK)

        lex = self.current()
        if not lex:
            return None

        children = []
        if published:
            children.append(TreeNode("PUBLISHED", lexeme=published))

        if lex.type == LexemeType.YEAR_NUM:
            year_lex = self.consume(LexemeType.YEAR_NUM)
            children.append(TreeNode("YEAR", lexeme=year_lex))
            if self.current() and self.current().type == LexemeType.YEAR_TERM:
                term = self.consume(LexemeType.YEAR_TERM)
                children.append(TreeNode("YEAR_TERM", lexeme=term))
            return TreeNode("YEAR_FILTER", children=children)

        elif lex.type == LexemeType.POST:
            post_lex = self.consume(LexemeType.POST)
            children.append(TreeNode("POST", lexeme=post_lex))
            year_lex = self.consume(LexemeType.YEAR_NUM)
            if not year_lex:
                self.position = start
                return None
            children.append(TreeNode("YEAR", lexeme=year_lex))
            if self.current() and self.current().type == LexemeType.YEAR_TERM:
                term = self.consume(LexemeType.YEAR_TERM)
                children.append(TreeNode("YEAR_TERM", lexeme=term))
            return TreeNode("YEAR_FILTER", children=children)

        elif lex.type == LexemeType.PRIOR:
            prior_lex = self.consume(LexemeType.PRIOR)
            children.append(TreeNode("PRIOR", lexeme=prior_lex))
            year_lex = self.consume(LexemeType.YEAR_NUM)
            if not year_lex:
                self.position = start
                return None
            children.append(TreeNode("YEAR", lexeme=year_lex))
            if self.current() and self.current().type == LexemeType.YEAR_TERM:
                term = self.consume(LexemeType.YEAR_TERM)
                children.append(TreeNode("YEAR_TERM", lexeme=term))
            return TreeNode("YEAR_FILTER", children=children)

        elif lex.type == LexemeType.INSIDE:
            inside_lex = self.consume(LexemeType.INSIDE)
            children.append(TreeNode("INSIDE", lexeme=inside_lex))
            year_lex = self.consume(LexemeType.YEAR_NUM)
            if not year_lex:
                self.position = start
                return None
            children.append(TreeNode("YEAR", lexeme=year_lex))
            if self.current() and self.current().type == LexemeType.YEAR_TERM:
                term = self.consume(LexemeType.YEAR_TERM)
                children.append(TreeNode("YEAR_TERM", lexeme=term))
            return TreeNode("YEAR_FILTER", children=children)

        return None

    def parse(self):
        tree = self.parse_root()
        if tree is None:
            if self.position < len(self.lexemes):
                lex = self.lexemes[self.position]
                raise SyntaxError(f"Ошибка на токене: {lex.value} (тип {lex.type})")
            else:
                raise SyntaxError("Неожиданный конец запроса")
        return tree