"""
LuaShield V4 - Real VM Bytecode Obfuscator
Real Lexer → Parser → AST → Compiler → VM
"""

import re
import random
import string
import hashlib
import time
import struct
import math
from typing import Optional, Dict, Any, List, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto


# ============== TOKEN TYPES ==============

class TokenType(Enum):
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NIL = auto()
    IDENTIFIER = auto()
    LOCAL = auto()
    FUNCTION = auto()
    END = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ELSEIF = auto()
    WHILE = auto()
    DO = auto()
    FOR = auto()
    IN = auto()
    REPEAT = auto()
    UNTIL = auto()
    RETURN = auto()
    BREAK = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    CARET = auto()
    HASH = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    ASSIGN = auto()
    CONCAT = auto()
    VARARG = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    DOT = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int


# ============== LEXER ==============

class Lexer:
    KEYWORDS = {
        'local': TokenType.LOCAL, 'function': TokenType.FUNCTION,
        'end': TokenType.END, 'if': TokenType.IF, 'then': TokenType.THEN,
        'else': TokenType.ELSE, 'elseif': TokenType.ELSEIF,
        'while': TokenType.WHILE, 'do': TokenType.DO, 'for': TokenType.FOR,
        'in': TokenType.IN, 'repeat': TokenType.REPEAT, 'until': TokenType.UNTIL,
        'return': TokenType.RETURN, 'break': TokenType.BREAK,
        'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
        'true': TokenType.TRUE, 'false': TokenType.FALSE, 'nil': TokenType.NIL,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        return self.source[pos] if pos < len(self.source) else '\0'
    
    def advance(self) -> str:
        char = self.peek()
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        while self.peek() in ' \t\r':
            self.advance()
    
    def skip_comment(self) -> bool:
        if self.peek() == '-' and self.peek(1) == '-':
            self.advance()
            self.advance()
            if self.peek() == '[' and self.peek(1) == '[':
                self.advance()
                self.advance()
                while not (self.peek() == ']' and self.peek(1) == ']'):
                    if self.peek() == '\0':
                        break
                    self.advance()
                self.advance()
                self.advance()
            else:
                while self.peek() != '\n' and self.peek() != '\0':
                    self.advance()
            return True
        return False
    
    def read_string(self, quote: str) -> str:
        result = []
        self.advance()
        while self.peek() != quote:
            if self.peek() == '\0':
                break
            if self.peek() == '\\':
                self.advance()
                c = self.advance()
                escape = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                result.append(escape.get(c, c))
            else:
                result.append(self.advance())
        self.advance()
        return ''.join(result)
    
    def read_long_string(self) -> str:
        self.advance()
        self.advance()
        result = []
        while not (self.peek() == ']' and self.peek(1) == ']'):
            if self.peek() == '\0':
                break
            result.append(self.advance())
        self.advance()
        self.advance()
        return ''.join(result)
    
    def read_number(self) -> float:
        result = []
        if self.peek() == '0' and self.peek(1) in 'xX':
            result.append(self.advance())
            result.append(self.advance())
            while self.peek() in '0123456789abcdefABCDEF':
                result.append(self.advance())
            return int(''.join(result), 16)
        while self.peek().isdigit():
            result.append(self.advance())
        if self.peek() == '.' and self.peek(1).isdigit():
            result.append(self.advance())
            while self.peek().isdigit():
                result.append(self.advance())
        if self.peek() in 'eE':
            result.append(self.advance())
            if self.peek() in '+-':
                result.append(self.advance())
            while self.peek().isdigit():
                result.append(self.advance())
        return float(''.join(result))
    
    def read_identifier(self) -> str:
        result = []
        while self.peek().isalnum() or self.peek() == '_':
            result.append(self.advance())
        return ''.join(result)
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            self.skip_whitespace()
            if self.peek() == '\0':
                break
            if self.skip_comment():
                continue
            if self.peek() == '\n':
                self.advance()
                continue
            
            line, col = self.line, self.column
            char = self.peek()
            
            if char in '"\'':
                self.tokens.append(Token(TokenType.STRING, self.read_string(char), line, col))
            elif char == '[' and self.peek(1) == '[':
                self.tokens.append(Token(TokenType.STRING, self.read_long_string(), line, col))
            elif char.isdigit():
                self.tokens.append(Token(TokenType.NUMBER, self.read_number(), line, col))
            elif char.isalpha() or char == '_':
                value = self.read_identifier()
                ttype = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(ttype, value, line, col))
            elif char == '+': self.advance(); self.tokens.append(Token(TokenType.PLUS, '+', line, col))
            elif char == '-': self.advance(); self.tokens.append(Token(TokenType.MINUS, '-', line, col))
            elif char == '*': self.advance(); self.tokens.append(Token(TokenType.STAR, '*', line, col))
            elif char == '/': self.advance(); self.tokens.append(Token(TokenType.SLASH, '/', line, col))
            elif char == '%': self.advance(); self.tokens.append(Token(TokenType.PERCENT, '%', line, col))
            elif char == '^': self.advance(); self.tokens.append(Token(TokenType.CARET, '^', line, col))
            elif char == '#': self.advance(); self.tokens.append(Token(TokenType.HASH, '#', line, col))
            elif char == '=' and self.peek(1) == '=': self.advance(); self.advance(); self.tokens.append(Token(TokenType.EQ, '==', line, col))
            elif char == '~' and self.peek(1) == '=': self.advance(); self.advance(); self.tokens.append(Token(TokenType.NE, '~=', line, col))
            elif char == '<' and self.peek(1) == '=': self.advance(); self.advance(); self.tokens.append(Token(TokenType.LE, '<=', line, col))
            elif char == '>' and self.peek(1) == '=': self.advance(); self.advance(); self.tokens.append(Token(TokenType.GE, '>=', line, col))
            elif char == '<': self.advance(); self.tokens.append(Token(TokenType.LT, '<', line, col))
            elif char == '>': self.advance(); self.tokens.append(Token(TokenType.GT, '>', line, col))
            elif char == '=': self.advance(); self.tokens.append(Token(TokenType.ASSIGN, '=', line, col))
            elif char == '.' and self.peek(1) == '.' and self.peek(2) == '.': self.advance(); self.advance(); self.advance(); self.tokens.append(Token(TokenType.VARARG, '...', line, col))
            elif char == '.' and self.peek(1) == '.': self.advance(); self.advance(); self.tokens.append(Token(TokenType.CONCAT, '..', line, col))
            elif char == '.': self.advance(); self.tokens.append(Token(TokenType.DOT, '.', line, col))
            elif char == '(': self.advance(); self.tokens.append(Token(TokenType.LPAREN, '(', line, col))
            elif char == ')': self.advance(); self.tokens.append(Token(TokenType.RPAREN, ')', line, col))
            elif char == '[': self.advance(); self.tokens.append(Token(TokenType.LBRACKET, '[', line, col))
            elif char == ']': self.advance(); self.tokens.append(Token(TokenType.RBRACKET, ']', line, col))
            elif char == '{': self.advance(); self.tokens.append(Token(TokenType.LBRACE, '{', line, col))
            elif char == '}': self.advance(); self.tokens.append(Token(TokenType.RBRACE, '}', line, col))
            elif char == ',': self.advance(); self.tokens.append(Token(TokenType.COMMA, ',', line, col))
            elif char == ';': self.advance(); self.tokens.append(Token(TokenType.SEMICOLON, ';', line, col))
            elif char == ':': self.advance(); self.tokens.append(Token(TokenType.COLON, ':', line, col))
            else: self.advance()
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


# ============== AST NODES ==============

class ASTNode: pass

@dataclass
class NumberNode(ASTNode): value: float
@dataclass
class StringNode(ASTNode): value: str
@dataclass
class BooleanNode(ASTNode): value: bool
@dataclass
class NilNode(ASTNode): pass
@dataclass
class IdentifierNode(ASTNode): name: str
@dataclass
class VarargNode(ASTNode): pass
@dataclass
class BinaryOpNode(ASTNode): op: str; left: ASTNode; right: ASTNode
@dataclass
class UnaryOpNode(ASTNode): op: str; operand: ASTNode
@dataclass
class TableNode(ASTNode): entries: List[Tuple[Optional[ASTNode], ASTNode]]
@dataclass
class IndexNode(ASTNode): table: ASTNode; index: ASTNode
@dataclass
class MemberNode(ASTNode): object: ASTNode; member: str
@dataclass
class CallNode(ASTNode): func: ASTNode; args: List[ASTNode]; is_method: bool = False
@dataclass
class MethodCallNode(ASTNode): object: ASTNode; method: str; args: List[ASTNode]
@dataclass
class FunctionNode(ASTNode): params: List[str]; body: List[ASTNode]; is_vararg: bool = False
@dataclass
class LocalNode(ASTNode): names: List[str]; values: List[ASTNode]
@dataclass
class AssignNode(ASTNode): targets: List[ASTNode]; values: List[ASTNode]
@dataclass
class IfNode(ASTNode): condition: ASTNode; then_body: List[ASTNode]; elseif_clauses: List[Tuple[ASTNode, List[ASTNode]]]; else_body: Optional[List[ASTNode]]
@dataclass
class WhileNode(ASTNode): condition: ASTNode; body: List[ASTNode]
@dataclass
class RepeatNode(ASTNode): body: List[ASTNode]; condition: ASTNode
@dataclass
class ForNumericNode(ASTNode): var: str; start: ASTNode; end: ASTNode; step: Optional[ASTNode]; body: List[ASTNode]
@dataclass
class ForGenericNode(ASTNode): vars: List[str]; iterators: List[ASTNode]; body: List[ASTNode]
@dataclass
class ReturnNode(ASTNode): values: List[ASTNode]
@dataclass
class BreakNode(ASTNode): pass
@dataclass
class BlockNode(ASTNode): statements: List[ASTNode]


# ============== PARSER ==============

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current(self) -> Token:
        return self.tokens[min(self.pos, len(self.tokens) - 1)]
    
    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]
    
    def match(self, *types: TokenType) -> bool:
        return self.current().type in types
    
    def advance(self) -> Token:
        token = self.current()
        self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        if not self.match(token_type):
            raise SyntaxError(f"Expected {token_type.name} at line {self.current().line}")
        return self.advance()
    
    def parse(self) -> BlockNode:
        return BlockNode(self.parse_block())
    
    def parse_block(self) -> List[ASTNode]:
        statements = []
        while not self.match(TokenType.EOF, TokenType.END, TokenType.ELSE, TokenType.ELSEIF, TokenType.UNTIL):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return statements
    
    def parse_statement(self) -> Optional[ASTNode]:
        while self.match(TokenType.SEMICOLON):
            self.advance()
        if self.match(TokenType.EOF):
            return None
        if self.match(TokenType.LOCAL):
            return self.parse_local()
        elif self.match(TokenType.FUNCTION):
            return self.parse_function_def()
        elif self.match(TokenType.IF):
            return self.parse_if()
        elif self.match(TokenType.WHILE):
            return self.parse_while()
        elif self.match(TokenType.REPEAT):
            return self.parse_repeat()
        elif self.match(TokenType.FOR):
            return self.parse_for()
        elif self.match(TokenType.RETURN):
            return self.parse_return()
        elif self.match(TokenType.BREAK):
            self.advance()
            return BreakNode()
        elif self.match(TokenType.DO):
            self.advance()
            body = self.parse_block()
            self.expect(TokenType.END)
            return BlockNode(body)
        else:
            return self.parse_expr_statement()
    
    def parse_local(self) -> ASTNode:
        self.expect(TokenType.LOCAL)
        if self.match(TokenType.FUNCTION):
            self.advance()
            name = self.expect(TokenType.IDENTIFIER).value
            func = self.parse_function_body()
            return LocalNode([name], [func])
        names = [self.expect(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            self.advance()
            names.append(self.expect(TokenType.IDENTIFIER).value)
        values = []
        if self.match(TokenType.ASSIGN):
            self.advance()
            values = self.parse_expr_list()
        return LocalNode(names, values)
    
    def parse_function_def(self) -> ASTNode:
        self.expect(TokenType.FUNCTION)
        name_parts = [self.expect(TokenType.IDENTIFIER).value]
        is_method = False
        while self.match(TokenType.DOT):
            self.advance()
            name_parts.append(self.expect(TokenType.IDENTIFIER).value)
        if self.match(TokenType.COLON):
            self.advance()
            name_parts.append(self.expect(TokenType.IDENTIFIER).value)
            is_method = True
        func = self.parse_function_body(is_method)
        if len(name_parts) == 1:
            return AssignNode([IdentifierNode(name_parts[0])], [func])
        target = IdentifierNode(name_parts[0])
        for part in name_parts[1:]:
            target = MemberNode(target, part)
        return AssignNode([target], [func])
    
    def parse_function_body(self, is_method: bool = False) -> FunctionNode:
        self.expect(TokenType.LPAREN)
        params = []
        is_vararg = False
        if is_method:
            params.append("self")
        if not self.match(TokenType.RPAREN):
            if self.match(TokenType.VARARG):
                self.advance()
                is_vararg = True
            else:
                params.append(self.expect(TokenType.IDENTIFIER).value)
                while self.match(TokenType.COMMA):
                    self.advance()
                    if self.match(TokenType.VARARG):
                        self.advance()
                        is_vararg = True
                        break
                    params.append(self.expect(TokenType.IDENTIFIER).value)
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        self.expect(TokenType.END)
        return FunctionNode(params, body, is_vararg)
    
    def parse_if(self) -> IfNode:
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.THEN)
        then_body = self.parse_block()
        elseif_clauses = []
        while self.match(TokenType.ELSEIF):
            self.advance()
            elseif_cond = self.parse_expression()
            self.expect(TokenType.THEN)
            elseif_body = self.parse_block()
            elseif_clauses.append((elseif_cond, elseif_body))
        else_body = None
        if self.match(TokenType.ELSE):
            self.advance()
            else_body = self.parse_block()
        self.expect(TokenType.END)
        return IfNode(condition, then_body, elseif_clauses, else_body)
    
    def parse_while(self) -> WhileNode:
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        self.expect(TokenType.DO)
        body = self.parse_block()
        self.expect(TokenType.END)
        return WhileNode(condition, body)
    
    def parse_repeat(self) -> RepeatNode:
        self.expect(TokenType.REPEAT)
        body = self.parse_block()
        self.expect(TokenType.UNTIL)
        condition = self.parse_expression()
        return RepeatNode(body, condition)
    
    def parse_for(self) -> ASTNode:
        self.expect(TokenType.FOR)
        first_var = self.expect(TokenType.IDENTIFIER).value
        if self.match(TokenType.ASSIGN):
            self.advance()
            start = self.parse_expression()
            self.expect(TokenType.COMMA)
            end = self.parse_expression()
            step = None
            if self.match(TokenType.COMMA):
                self.advance()
                step = self.parse_expression()
            self.expect(TokenType.DO)
            body = self.parse_block()
            self.expect(TokenType.END)
            return ForNumericNode(first_var, start, end, step, body)
        else:
            vars = [first_var]
            while self.match(TokenType.COMMA):
                self.advance()
                vars.append(self.expect(TokenType.IDENTIFIER).value)
            self.expect(TokenType.IN)
            iterators = self.parse_expr_list()
            self.expect(TokenType.DO)
            body = self.parse_block()
            self.expect(TokenType.END)
            return ForGenericNode(vars, iterators, body)
    
    def parse_return(self) -> ReturnNode:
        self.expect(TokenType.RETURN)
        values = []
        if not self.match(TokenType.END, TokenType.ELSE, TokenType.ELSEIF, TokenType.UNTIL, TokenType.EOF, TokenType.SEMICOLON):
            values = self.parse_expr_list()
        return ReturnNode(values)
    
    def parse_expr_statement(self) -> ASTNode:
        expr = self.parse_suffixed_expr()
        if self.match(TokenType.ASSIGN) or self.match(TokenType.COMMA):
            targets = [expr]
            while self.match(TokenType.COMMA):
                self.advance()
                targets.append(self.parse_suffixed_expr())
            self.expect(TokenType.ASSIGN)
            values = self.parse_expr_list()
            return AssignNode(targets, values)
        return expr
    
    def parse_expr_list(self) -> List[ASTNode]:
        exprs = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            self.advance()
            exprs.append(self.parse_expression())
        return exprs
    
    def parse_expression(self) -> ASTNode:
        return self.parse_or_expr()
    
    def parse_or_expr(self) -> ASTNode:
        left = self.parse_and_expr()
        while self.match(TokenType.OR):
            self.advance()
            left = BinaryOpNode('or', left, self.parse_and_expr())
        return left
    
    def parse_and_expr(self) -> ASTNode:
        left = self.parse_comparison()
        while self.match(TokenType.AND):
            self.advance()
            left = BinaryOpNode('and', left, self.parse_comparison())
        return left
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_concat()
        while self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE, TokenType.EQ, TokenType.NE):
            op = self.advance().value
            left = BinaryOpNode(op, left, self.parse_concat())
        return left
    
    def parse_concat(self) -> ASTNode:
        left = self.parse_additive()
        if self.match(TokenType.CONCAT):
            self.advance()
            return BinaryOpNode('..', left, self.parse_concat())
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            left = BinaryOpNode(op, left, self.parse_multiplicative())
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_unary()
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            left = BinaryOpNode(op, left, self.parse_unary())
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT):
            self.advance()
            return UnaryOpNode('not', self.parse_unary())
        elif self.match(TokenType.MINUS):
            self.advance()
            return UnaryOpNode('-', self.parse_unary())
        elif self.match(TokenType.HASH):
            self.advance()
            return UnaryOpNode('#', self.parse_unary())
        return self.parse_power()
    
    def parse_power(self) -> ASTNode:
        left = self.parse_suffixed_expr()
        if self.match(TokenType.CARET):
            self.advance()
            return BinaryOpNode('^', left, self.parse_unary())
        return left
    
    def parse_suffixed_expr(self) -> ASTNode:
        primary = self.parse_primary()
        while True:
            if self.match(TokenType.DOT):
                self.advance()
                primary = MemberNode(primary, self.expect(TokenType.IDENTIFIER).value)
            elif self.match(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                primary = IndexNode(primary, index)
            elif self.match(TokenType.COLON):
                self.advance()
                method = self.expect(TokenType.IDENTIFIER).value
                args = self.parse_call_args()
                primary = MethodCallNode(primary, method, args)
            elif self.match(TokenType.LPAREN, TokenType.LBRACE, TokenType.STRING):
                primary = CallNode(primary, self.parse_call_args())
            else:
                break
        return primary
    
    def parse_call_args(self) -> List[ASTNode]:
        if self.match(TokenType.LPAREN):
            self.advance()
            args = [] if self.match(TokenType.RPAREN) else self.parse_expr_list()
            self.expect(TokenType.RPAREN)
            return args
        elif self.match(TokenType.LBRACE):
            return [self.parse_table()]
        elif self.match(TokenType.STRING):
            return [StringNode(self.advance().value)]
        return []
    
    def parse_primary(self) -> ASTNode:
        if self.match(TokenType.NUMBER):
            return NumberNode(self.advance().value)
        elif self.match(TokenType.STRING):
            return StringNode(self.advance().value)
        elif self.match(TokenType.TRUE):
            self.advance()
            return BooleanNode(True)
        elif self.match(TokenType.FALSE):
            self.advance()
            return BooleanNode(False)
        elif self.match(TokenType.NIL):
            self.advance()
            return NilNode()
        elif self.match(TokenType.VARARG):
            self.advance()
            return VarargNode()
        elif self.match(TokenType.IDENTIFIER):
            return IdentifierNode(self.advance().value)
        elif self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        elif self.match(TokenType.LBRACE):
            return self.parse_table()
        elif self.match(TokenType.FUNCTION):
            self.advance()
            return self.parse_function_body()
        raise SyntaxError(f"Unexpected token at line {self.current().line}")
    
    def parse_table(self) -> TableNode:
        self.expect(TokenType.LBRACE)
        entries = []
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.LBRACKET):
                self.advance()
                key = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                entries.append((key, value))
            elif self.match(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
                key = StringNode(self.advance().value)
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                entries.append((key, value))
            else:
                entries.append((None, self.parse_expression()))
            if self.match(TokenType.COMMA, TokenType.SEMICOLON):
                self.advance()
            else:
                break
        self.expect(TokenType.RBRACE)
        return TableNode(entries)


# ============== OPCODES ==============

class Opcode(Enum):
    LOADK = 0x01
    LOADNIL = 0x02
    LOADBOOL = 0x03
    MOVE = 0x04
    GETGLOBAL = 0x05
    SETGLOBAL = 0x06
    GETUPVAL = 0x07
    SETUPVAL = 0x08
    NEWTABLE = 0x09
    GETTABLE = 0x0A
    SETTABLE = 0x0B
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    POW = 0x15
    UNM = 0x16
    BAND = 0x20
    BOR = 0x21
    BXOR = 0x22
    BNOT = 0x23
    SHL = 0x24
    SHR = 0x25
    EQ = 0x30
    LT = 0x31
    LE = 0x32
    NOT = 0x33
    CONCAT = 0x40
    LEN = 0x41
    JMP = 0x50
    JMPIF = 0x51
    JMPIFNOT = 0x52
    FORPREP = 0x60
    FORLOOP = 0x61
    TFORCALL = 0x62
    TFORLOOP = 0x63
    CALL = 0x70
    TAILCALL = 0x71
    RETURN = 0x72
    CLOSURE = 0x73
    VARARG = 0x74
    SELF = 0x75
    NOP = 0xFE
    HALT = 0xFF


@dataclass
class Instruction:
    opcode: Opcode
    a: int = 0
    b: int = 0
    c: int = 0
    sbx: int = 0


@dataclass
class Prototype:
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    prototypes: List['Prototype'] = field(default_factory=list)
    num_params: int = 0
    is_vararg: bool = False
    max_stack: int = 0
    upvalues: List[str] = field(default_factory=list)
    locals: List[str] = field(default_factory=list)


# ============== COMPILER ==============

class Compiler:
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.proto = Prototype()
        self.scope_stack: List[Dict[str, int]] = [{}]
        self.register_top = 0
        self.max_register = 0
        self.break_jumps: List[List[int]] = [[]]
        self.opcode_map = self._generate_opcode_map()
    
    def _generate_opcode_map(self) -> Dict[Opcode, int]:
        opcodes = list(Opcode)
        values = list(range(1, len(opcodes) + 1))
        random.shuffle(values)
        return {op: val for op, val in zip(opcodes, values)}
    
    def compile(self, ast: BlockNode) -> Prototype:
        self.compile_block(ast.statements)
        self.emit(Opcode.RETURN, 0, 1)
        self.proto.max_stack = self.max_register + 1
        return self.proto
    
    def emit(self, opcode: Opcode, a: int = 0, b: int = 0, c: int = 0) -> int:
        self.proto.instructions.append(Instruction(opcode, a, b, c))
        return len(self.proto.instructions) - 1
    
    def emit_jump(self, opcode: Opcode, a: int = 0) -> int:
        return self.emit(opcode, a, 0, 0)
    
    def patch_jump(self, idx: int, target: int):
        self.proto.instructions[idx].sbx = target - idx - 1
    
    def current_pc(self) -> int:
        return len(self.proto.instructions)
    
    def alloc_register(self, count: int = 1) -> int:
        reg = self.register_top
        self.register_top += count
        self.max_register = max(self.max_register, self.register_top)
        return reg
    
    def free_register(self, count: int = 1):
        self.register_top -= count
    
    def add_constant(self, value: Any) -> int:
        if value in self.proto.constants:
            return self.proto.constants.index(value)
        self.proto.constants.append(value)
        return len(self.proto.constants) - 1
    
    def push_scope(self):
        self.scope_stack.append({})
    
    def pop_scope(self):
        scope = self.scope_stack.pop()
        self.free_register(len(scope))
    
    def declare_local(self, name: str) -> int:
        reg = self.alloc_register()
        self.scope_stack[-1][name] = reg
        return reg
    
    def resolve_local(self, name: str) -> Optional[int]:
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None
    
    def compile_block(self, statements: List[ASTNode]):
        for stmt in statements:
            self.compile_statement(stmt)
    
    def compile_statement(self, node: ASTNode):
        if isinstance(node, LocalNode):
            self.compile_local(node)
        elif isinstance(node, AssignNode):
            self.compile_assign(node)
        elif isinstance(node, IfNode):
            self.compile_if(node)
        elif isinstance(node, WhileNode):
            self.compile_while(node)
        elif isinstance(node, RepeatNode):
            self.compile_repeat(node)
        elif isinstance(node, ForNumericNode):
            self.compile_for_numeric(node)
        elif isinstance(node, ForGenericNode):
            self.compile_for_generic(node)
        elif isinstance(node, ReturnNode):
            self.compile_return(node)
        elif isinstance(node, BreakNode):
            jump = self.emit_jump(Opcode.JMP)
            if self.break_jumps:
                self.break_jumps[-1].append(jump)
        elif isinstance(node, BlockNode):
            self.push_scope()
            self.compile_block(node.statements)
            self.pop_scope()
        elif isinstance(node, (CallNode, MethodCallNode)):
            reg = self.alloc_register()
            self.compile_expression(node, reg)
            self.free_register()
        else:
            reg = self.alloc_register()
            self.compile_expression(node, reg)
            self.free_register()
    
    def compile_local(self, node: LocalNode):
        regs = [self.declare_local(name) for name in node.names]
        for i, value in enumerate(node.values):
            if i < len(regs):
                self.compile_expression(value, regs[i])
        for i in range(len(node.values), len(regs)):
            self.emit(Opcode.LOADNIL, regs[i])
    
    def compile_assign(self, node: AssignNode):
        temp_regs = []
        for value in node.values:
            reg = self.alloc_register()
            temp_regs.append(reg)
            self.compile_expression(value, reg)
        
        for i, target in enumerate(node.targets):
            value_reg = temp_regs[i] if i < len(temp_regs) else -1
            if isinstance(target, IdentifierNode):
                local_reg = self.resolve_local(target.name)
                if local_reg is not None:
                    if value_reg >= 0:
                        self.emit(Opcode.MOVE, local_reg, value_reg)
                    else:
                        self.emit(Opcode.LOADNIL, local_reg)
                else:
                    const_idx = self.add_constant(target.name)
                    if value_reg >= 0:
                        self.emit(Opcode.SETGLOBAL, value_reg, const_idx)
            elif isinstance(target, MemberNode):
                obj_reg = self.alloc_register()
                self.compile_expression(target.object, obj_reg)
                key_const = self.add_constant(target.member)
                if value_reg >= 0:
                    self.emit(Opcode.SETTABLE, obj_reg, key_const, value_reg)
                self.free_register()
            elif isinstance(target, IndexNode):
                obj_reg = self.alloc_register()
                key_reg = self.alloc_register()
                self.compile_expression(target.table, obj_reg)
                self.compile_expression(target.index, key_reg)
                if value_reg >= 0:
                    self.emit(Opcode.SETTABLE, obj_reg, key_reg, value_reg)
                self.free_register(2)
        
        self.free_register(len(temp_regs))
    
    def compile_if(self, node: IfNode):
        cond_reg = self.alloc_register()
        self.compile_expression(node.condition, cond_reg)
        false_jump = self.emit_jump(Opcode.JMPIFNOT, cond_reg)
        self.free_register()
        
        self.push_scope()
        self.compile_block(node.then_body)
        self.pop_scope()
        
        end_jumps = []
        for elseif_cond, elseif_body in node.elseif_clauses:
            end_jumps.append(self.emit_jump(Opcode.JMP))
            self.patch_jump(false_jump, self.current_pc())
            cond_reg = self.alloc_register()
            self.compile_expression(elseif_cond, cond_reg)
            false_jump = self.emit_jump(Opcode.JMPIFNOT, cond_reg)
            self.free_register()
            self.push_scope()
            self.compile_block(elseif_body)
            self.pop_scope()
        
        if node.else_body:
            end_jumps.append(self.emit_jump(Opcode.JMP))
            self.patch_jump(false_jump, self.current_pc())
            self.push_scope()
            self.compile_block(node.else_body)
            self.pop_scope()
        else:
            self.patch_jump(false_jump, self.current_pc())
        
        for jump in end_jumps:
            self.patch_jump(jump, self.current_pc())
    
    def compile_while(self, node: WhileNode):
        loop_start = self.current_pc()
        self.break_jumps.append([])
        cond_reg = self.alloc_register()
        self.compile_expression(node.condition, cond_reg)
        exit_jump = self.emit_jump(Opcode.JMPIFNOT, cond_reg)
        self.free_register()
        self.push_scope()
        self.compile_block(node.body)
        self.pop_scope()
        self.emit(Opcode.JMP, 0, 0, 0)
        self.patch_jump(self.current_pc() - 1, loop_start)
        self.patch_jump(exit_jump, self.current_pc())
        for brk in self.break_jumps.pop():
            self.patch_jump(brk, self.current_pc())
    
    def compile_repeat(self, node: RepeatNode):
        loop_start = self.current_pc()
        self.break_jumps.append([])
        self.push_scope()
        self.compile_block(node.body)
        cond_reg = self.alloc_register()
        self.compile_expression(node.condition, cond_reg)
        self.emit(Opcode.JMPIFNOT, cond_reg, 0, 0)
        self.patch_jump(self.current_pc() - 1, loop_start)
        self.free_register()
        self.pop_scope()
        for brk in self.break_jumps.pop():
            self.patch_jump(brk, self.current_pc())
    
    def compile_for_numeric(self, node: ForNumericNode):
        self.push_scope()
        base_reg = self.alloc_register(4)
        self.compile_expression(node.start, base_reg)
        self.compile_expression(node.end, base_reg + 1)
        if node.step:
            self.compile_expression(node.step, base_reg + 2)
        else:
            self.emit(Opcode.LOADK, base_reg + 2, self.add_constant(1))
        prep_jump = self.emit(Opcode.FORPREP, base_reg, 0, 0)
        loop_start = self.current_pc()
        self.break_jumps.append([])
        self.scope_stack[-1][node.var] = base_reg + 3
        self.compile_block(node.body)
        self.patch_jump(prep_jump, self.current_pc())
        self.emit(Opcode.FORLOOP, base_reg, 0, 0)
        self.patch_jump(self.current_pc() - 1, loop_start)
        for brk in self.break_jumps.pop():
            self.patch_jump(brk, self.current_pc())
        self.pop_scope()
    
    def compile_for_generic(self, node: ForGenericNode):
        self.push_scope()
        base_reg = self.alloc_register(3 + len(node.vars))
        for i, it in enumerate(node.iterators[:3]):
            self.compile_expression(it, base_reg + i)
        prep_jump = self.emit_jump(Opcode.JMP)
        loop_start = self.current_pc()
        self.break_jumps.append([])
        for i, var in enumerate(node.vars):
            self.scope_stack[-1][var] = base_reg + 3 + i
        self.compile_block(node.body)
        self.patch_jump(prep_jump, self.current_pc())
        self.emit(Opcode.TFORCALL, base_reg, 0, len(node.vars))
        self.emit(Opcode.TFORLOOP, base_reg + 2, 0, 0)
        self.patch_jump(self.current_pc() - 1, loop_start)
        for brk in self.break_jumps.pop():
            self.patch_jump(brk, self.current_pc())
        self.pop_scope()
    
    def compile_return(self, node: ReturnNode):
        if not node.values:
            self.emit(Opcode.RETURN, 0, 1)
        else:
            base_reg = self.alloc_register(len(node.values))
            for i, value in enumerate(node.values):
                self.compile_expression(value, base_reg + i)
            self.emit(Opcode.RETURN, base_reg, len(node.values) + 1)
            self.free_register(len(node.values))
    
    def compile_expression(self, node: ASTNode, target_reg: int):
        if isinstance(node, NumberNode):
            self.emit(Opcode.LOADK, target_reg, self.add_constant(node.value))
        elif isinstance(node, StringNode):
            self.emit(Opcode.LOADK, target_reg, self.add_constant(node.value))
        elif isinstance(node, BooleanNode):
            self.emit(Opcode.LOADBOOL, target_reg, 1 if node.value else 0)
        elif isinstance(node, NilNode):
            self.emit(Opcode.LOADNIL, target_reg)
        elif isinstance(node, IdentifierNode):
            local_reg = self.resolve_local(node.name)
            if local_reg is not None:
                self.emit(Opcode.MOVE, target_reg, local_reg)
            else:
                self.emit(Opcode.GETGLOBAL, target_reg, self.add_constant(node.name))
        elif isinstance(node, BinaryOpNode):
            self.compile_binary_op(node, target_reg)
        elif isinstance(node, UnaryOpNode):
            self.compile_unary_op(node, target_reg)
        elif isinstance(node, TableNode):
            self.compile_table(node, target_reg)
        elif isinstance(node, MemberNode):
            obj_reg = self.alloc_register()
            self.compile_expression(node.object, obj_reg)
            self.emit(Opcode.GETTABLE, target_reg, obj_reg, self.add_constant(node.member))
            self.free_register()
        elif isinstance(node, IndexNode):
            obj_reg = self.alloc_register()
            key_reg = self.alloc_register()
            self.compile_expression(node.table, obj_reg)
            self.compile_expression(node.index, key_reg)
            self.emit(Opcode.GETTABLE, target_reg, obj_reg, key_reg)
            self.free_register(2)
        elif isinstance(node, CallNode):
            self.compile_call(node, target_reg)
        elif isinstance(node, MethodCallNode):
            self.compile_method_call(node, target_reg)
        elif isinstance(node, FunctionNode):
            self.compile_function(node, target_reg)
        elif isinstance(node, VarargNode):
            self.emit(Opcode.VARARG, target_reg, 0)
    
    def compile_binary_op(self, node: BinaryOpNode, target_reg: int):
        left_reg = self.alloc_register()
        right_reg = self.alloc_register()
        self.compile_expression(node.left, left_reg)
        self.compile_expression(node.right, right_reg)
        op_map = {'+': Opcode.ADD, '-': Opcode.SUB, '*': Opcode.MUL, '/': Opcode.DIV,
                  '%': Opcode.MOD, '^': Opcode.POW, '..': Opcode.CONCAT,
                  '==': Opcode.EQ, '~=': Opcode.EQ, '<': Opcode.LT, '<=': Opcode.LE,
                  '>': Opcode.LT, '>=': Opcode.LE}
        opcode = op_map.get(node.op)
        if node.op in ('>', '>='):
            self.emit(opcode, target_reg, right_reg, left_reg)
        else:
            self.emit(opcode, target_reg, left_reg, right_reg)
        if node.op == '~=':
            self.emit(Opcode.NOT, target_reg, target_reg)
        self.free_register(2)
    
    def compile_unary_op(self, node: UnaryOpNode, target_reg: int):
        self.compile_expression(node.operand, target_reg)
        op_map = {'-': Opcode.UNM, 'not': Opcode.NOT, '#': Opcode.LEN}
        if node.op in op_map:
            self.emit(op_map[node.op], target_reg, target_reg)
    
    def compile_table(self, node: TableNode, target_reg: int):
        self.emit(Opcode.NEWTABLE, target_reg, len(node.entries))
        array_idx = 1
        for key, value in node.entries:
            value_reg = self.alloc_register()
            self.compile_expression(value, value_reg)
            if key is None:
                self.emit(Opcode.SETTABLE, target_reg, self.add_constant(array_idx), value_reg)
                array_idx += 1
            else:
                key_reg = self.alloc_register()
                self.compile_expression(key, key_reg)
                self.emit(Opcode.SETTABLE, target_reg, key_reg, value_reg)
                self.free_register()
            self.free_register()
    
    def compile_call(self, node: CallNode, target_reg: int):
        func_reg = self.alloc_register()
        self.compile_expression(node.func, func_reg)
        for arg in node.args:
            arg_reg = self.alloc_register()
            self.compile_expression(arg, arg_reg)
        self.emit(Opcode.CALL, target_reg, func_reg, len(node.args) + 1)
        self.free_register(len(node.args) + 1)
    
    def compile_method_call(self, node: MethodCallNode, target_reg: int):
        obj_reg = self.alloc_register()
        self.compile_expression(node.object, obj_reg)
        self.emit(Opcode.SELF, target_reg, obj_reg, self.add_constant(node.method))
        for arg in node.args:
            arg_reg = self.alloc_register()
            self.compile_expression(arg, arg_reg)
        self.emit(Opcode.CALL, target_reg, target_reg, len(node.args) + 2)
        self.free_register(len(node.args) + 1)
    
    def compile_function(self, node: FunctionNode, target_reg: int):
        func_compiler = Compiler()
        func_compiler.proto.num_params = len(node.params)
        func_compiler.proto.is_vararg = node.is_vararg
        func_compiler.push_scope()
        for param in node.params:
            func_compiler.declare_local(param)
        func_compiler.compile_block(node.body)
        func_compiler.emit(Opcode.RETURN, 0, 1)
        func_compiler.pop_scope()
        func_compiler.proto.max_stack = func_compiler.max_register + 1
        proto_idx = len(self.proto.prototypes)
        self.proto.prototypes.append(func_compiler.proto)
        self.emit(Opcode.CLOSURE, target_reg, proto_idx)


# ============== SERIALIZER ==============

class BytecodeSerializer:
    def __init__(self, seed: int):
        random.seed(seed)
        self.keys = [random.randint(0, 0xFFFFFFFF) for _ in range(8)]
    
    def serialize(self, proto: Prototype) -> bytes:
        data = bytearray(b'LSVM')
        data.extend(struct.pack('<I', len(self.keys)))
        for key in self.keys:
            data.extend(struct.pack('<I', key))
        self._serialize_proto(data, proto)
        encrypted = self._encrypt(bytes(data[4:]))
        return bytes(data[:4]) + encrypted
    
    def _serialize_proto(self, data: bytearray, proto: Prototype):
        data.extend(struct.pack('<BBBB', proto.num_params, 1 if proto.is_vararg else 0, proto.max_stack, len(proto.upvalues)))
        data.extend(struct.pack('<I', len(proto.instructions)))
        for instr in proto.instructions:
            packed = (instr.opcode.value & 0xFF) << 24 | (instr.a & 0xFF) << 16 | (instr.b & 0xFF) << 8 | (instr.c & 0xFF)
            packed ^= self.keys[len(data) % len(self.keys)]
            data.extend(struct.pack('<I', packed))
        data.extend(struct.pack('<I', len(proto.constants)))
        for const in proto.constants:
            if const is None:
                data.append(0)
            elif isinstance(const, bool):
                data.append(1)
                data.append(1 if const else 0)
            elif isinstance(const, (int, float)):
                data.append(2)
                data.extend(struct.pack('<d', float(const)))
            elif isinstance(const, str):
                data.append(3)
                encoded = const.encode('utf-8')
                data.extend(struct.pack('<I', len(encoded)))
                data.extend(encoded)
        data.extend(struct.pack('<I', len(proto.prototypes)))
        for child in proto.prototypes:
            self._serialize_proto(data, child)
    
    def _encrypt(self, data: bytes) -> bytes:
        result = bytearray(data)
        for i in range(len(result)):
            result[i] ^= self.keys[i % len(self.keys)] & 0xFF
        for i in range(len(result)):
            result[i] = (result[i] + i * 7) & 0xFF
        for i in range(0, len(result) - 1, 2):
            result[i], result[i + 1] = result[i + 1], result[i]
        return bytes(result)


# ============== NAME GENERATOR ==============

class NameGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.counters = {}
        self.used: Set[str] = set()
        self.reserved = {'if', 'then', 'end', 'local', 'function', 'return', 'and', 'or', 'not', 'nil', 'true', 'false', 'for', 'while', 'do', 'in', 'else', 'elseif', 'repeat', 'until', 'break'}
    
    def generate(self, style: str = "mixed") -> str:
        while True:
            name = self._gen(style)
            if name not in self.used and name not in self.reserved:
                self.used.add(name)
                return name
    
    def _gen(self, style: str) -> str:
        self.counters[style] = self.counters.get(style, 0) + 1
        n = self.counters[style]
        if style == "leet":
            return random.choice(['O', 'o', '_']) + ''.join(random.choices(['O', 'o', '0', '_'], k=random.randint(8, 12))) + f'{n:X}'
        elif style == "il":
            return random.choice(['l', 'I', '_']) + ''.join(random.choices(['l', 'I', '1', '_'], k=random.randint(8, 12))) + f'{n:X}'
        elif style == "camel":
            return ''.join(random.choices(string.ascii_letters, k=random.randint(10, 14))) + f'{n:X}'
        elif style == "underscore":
            return '_' + ''.join(random.choices(string.ascii_letters, k=random.randint(8, 12))) + f'{n:X}'
        return self._gen(random.choice(["leet", "il", "camel", "underscore"]))


# ============== VM RUNTIME GENERATOR ==============

class VMRuntimeGenerator:
    def __init__(self, seed: int):
        random.seed(seed)
        self.name_gen = NameGenerator(seed)
    
    def generate(self, serialized: bytes, opcode_map: Dict[Opcode, int]) -> str:
        v = {k: self.name_gen.generate(random.choice(["leet", "il", "camel"])) for k in 
             ['bc', 'keys', 'decrypt', 'deser', 'run', 'handlers', 'regs', 'env', 'pc']}
        
        bc_array = ','.join(str(b) for b in serialized)
        
        return f'''
local {v['bc']}={{{bc_array}}}
local {v['keys']}={{}}
local _kp=5
for _ki=1,8 do local _kv=0;for _kb=0,3 do _kv=_kv+{v['bc']}[_kp]*math.pow(256,_kb);_kp=_kp+1 end;{v['keys']}[_ki]=_kv end
local {v['decrypt']}=function(d,s)local r={{}};for i=s,#d do local b=d[i];b=bit32.bxor(b,{v['keys']}[(i-s)%8+1]%256);b=(b-(i-s)*7)%256;r[#r+1]=b end;for i=1,#r-1,2 do r[i],r[i+1]=r[i+1],r[i]end;return r end
local {v['deser']}=function(d)local p=1;local function rb()local v=d[p];p=p+1;return v end;local function ri()return rb()+rb()*256+rb()*65536+rb()*16777216 end;local function rd()local b={{}};for i=1,8 do b[i]=rb()end;local s=bit32.band(b[8],0x80)~=0 and-1 or 1;b[8]=bit32.band(b[8],0x7F);local e=bit32.lshift(b[8],4)+bit32.rshift(b[7],4)-1023;local m=1;for i=1,52 do local bi=math.floor((52-i)/8)+1;local bt=(52-i)%8;if bit32.band(b[bi],bit32.lshift(1,bt))~=0 then m=m+math.pow(2,i-52)end end;return s*m*math.pow(2,e)end;local function rs()local l=ri();local c={{}};for i=1,l do c[i]=string.char(rb())end;return table.concat(c)end;local function rp()local pr={{}};pr.np=rb();pr.va=rb()==1;pr.st=rb();pr.uv=rb();local ni=ri();pr.code={{}};for i=1,ni do local pk=ri();pk=bit32.bxor(pk,{v['keys']}[(p-1)%8+1]);pr.code[i]={{op=bit32.band(bit32.rshift(pk,24),0xFF),a=bit32.band(bit32.rshift(pk,16),0xFF),b=bit32.band(bit32.rshift(pk,8),0xFF),c=bit32.band(pk,0xFF)}}end;local nc=ri();pr.k={{}};for i=1,nc do local t=rb();if t==0 then pr.k[i]=nil elseif t==1 then pr.k[i]=rb()==1 elseif t==2 then pr.k[i]=rd()elseif t==3 then pr.k[i]=rs()end end;local np=ri();pr.p={{}};for i=1,np do pr.p[i]=rp()end;return pr end;return rp()end
local {v['handlers']}={{}}
{v['handlers']}[{opcode_map[Opcode.LOADK]}]=function(i,r,k)r[i.a]=k[i.b+1]end
{v['handlers']}[{opcode_map[Opcode.LOADNIL]}]=function(i,r,k)r[i.a]=nil end
{v['handlers']}[{opcode_map[Opcode.LOADBOOL]}]=function(i,r,k)r[i.a]=i.b==1 end
{v['handlers']}[{opcode_map[Opcode.MOVE]}]=function(i,r,k)r[i.a]=r[i.b]end
{v['handlers']}[{opcode_map[Opcode.GETGLOBAL]}]=function(i,r,k,e)r[i.a]=e[k[i.b+1]]end
{v['handlers']}[{opcode_map[Opcode.SETGLOBAL]}]=function(i,r,k,e)e[k[i.b+1]]=r[i.a]end
{v['handlers']}[{opcode_map[Opcode.NEWTABLE]}]=function(i,r,k)r[i.a]={{}}end
{v['handlers']}[{opcode_map[Opcode.GETTABLE]}]=function(i,r,k)r[i.a]=r[i.b][i.c<128 and r[i.c]or k[i.c-127]]end
{v['handlers']}[{opcode_map[Opcode.SETTABLE]}]=function(i,r,k)r[i.a][i.b<128 and r[i.b]or k[i.b-127]]=i.c<128 and r[i.c]or k[i.c-127]end
{v['handlers']}[{opcode_map[Opcode.ADD]}]=function(i,r,k)r[i.a]=r[i.b]+r[i.c]end
{v['handlers']}[{opcode_map[Opcode.SUB]}]=function(i,r,k)r[i.a]=r[i.b]-r[i.c]end
{v['handlers']}[{opcode_map[Opcode.MUL]}]=function(i,r,k)r[i.a]=r[i.b]*r[i.c]end
{v['handlers']}[{opcode_map[Opcode.DIV]}]=function(i,r,k)r[i.a]=r[i.b]/r[i.c]end
{v['handlers']}[{opcode_map[Opcode.MOD]}]=function(i,r,k)r[i.a]=r[i.b]%r[i.c]end
{v['handlers']}[{opcode_map[Opcode.POW]}]=function(i,r,k)r[i.a]=r[i.b]^r[i.c]end
{v['handlers']}[{opcode_map[Opcode.UNM]}]=function(i,r,k)r[i.a]=-r[i.b]end
{v['handlers']}[{opcode_map[Opcode.NOT]}]=function(i,r,k)r[i.a]=not r[i.b]end
{v['handlers']}[{opcode_map[Opcode.LEN]}]=function(i,r,k)r[i.a]=#r[i.b]end
{v['handlers']}[{opcode_map[Opcode.CONCAT]}]=function(i,r,k)r[i.a]=tostring(r[i.b])..tostring(r[i.c])end
{v['handlers']}[{opcode_map[Opcode.EQ]}]=function(i,r,k)r[i.a]=r[i.b]==r[i.c]end
{v['handlers']}[{opcode_map[Opcode.LT]}]=function(i,r,k)r[i.a]=r[i.b]<r[i.c]end
{v['handlers']}[{opcode_map[Opcode.LE]}]=function(i,r,k)r[i.a]=r[i.b]<=r[i.c]end
{v['handlers']}[{opcode_map[Opcode.CALL]}]=function(i,r,k)local f=r[i.b];local a={{}};for j=1,i.c-1 do a[j]=r[i.b+j]end;local t={{f(unpack(a))}};r[i.a]=t[1]end
{v['handlers']}[{opcode_map[Opcode.RETURN]}]=function(i,r,k)return r[i.a]end
{v['handlers']}[{opcode_map[Opcode.SELF]}]=function(i,r,k)r[i.a+1]=r[i.b];r[i.a]=r[i.b][k[i.c+1]]end
local {v['run']};{v['run']}=function(pr,{v['env']},...)
local {v['regs']}={{}};local args={{...}};for i=1,pr.np do {v['regs']}[i-1]=args[i]end
local {v['pc']}=1
while {v['pc']}<=#pr.code do
local ins=pr.code[{v['pc']}];{v['pc']}={v['pc']}+1
local h={v['handlers']}[ins.op]
if h then local res=h(ins,{v['regs']},pr.k,{v['env']},pr.p,{v['run']})
if ins.op=={opcode_map[Opcode.RETURN]} then return res end
end
end
end
local _d={v['decrypt']}({v['bc']},37)
local _p={v['deser']}(_d)
local _e=setmetatable({{}},{{__index=_G}})
pcall(function(){v['run']}(_p,_e)end)
'''


# ============== RESULT CLASS ==============

@dataclass
class ObfuscationResult:
    success: bool = False
    code: str = ""
    error: Optional[str] = None
    original_size: int = 0
    obfuscated_size: int = 0
    bytecode_size: int = 0
    instruction_count: int = 0
    constant_count: int = 0
    time_ms: float = 0


# ============== MAIN OBFUSCATOR CLASS ==============

class RealVMObfuscator:
    """Real VM-based Lua obfuscator with actual bytecode compilation"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time() * 1000) % 1000000
        random.seed(self.seed)
    
    def obfuscate(self, code: str) -> ObfuscationResult:
        result = ObfuscationResult()
        start = time.time()
        
        try:
            result.original_size = len(code)
            
            # Pipeline: Lexer → Parser → Compiler → Serializer → Generator
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            
            parser = Parser(tokens)
            ast = parser.parse()
            
            compiler = Compiler(self.seed)
            proto = compiler.compile(ast)
            
            result.instruction_count = len(proto.instructions)
            result.constant_count = len(proto.constants)
            
            serializer = BytecodeSerializer(self.seed)
            bytecode = serializer.serialize(proto)
            result.bytecode_size = len(bytecode)
            
            runtime_gen = VMRuntimeGenerator(self.seed)
            output = runtime_gen.generate(bytecode, compiler.opcode_map)
            output = self._minify(output)
            
            result.success = True
            result.code = output
            result.obfuscated_size = len(output)
            result.time_ms = round((time.time() - start) * 1000, 2)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        return result
    
    def _minify(self, code: str) -> str:
        code = re.sub(r'--[^\n]*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'[ \t]+', ' ', code)
        return code.strip()


# ============== EXPORTS ==============

__all__ = ['RealVMObfuscator', 'ObfuscationResult', 'Lexer', 'Parser', 'Compiler']
