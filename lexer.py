"""
Luau Lexer - Tokenizer untuk Roblox Lua
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NIL = auto()
    VARARG = auto()
    
    # Identifiers & Keywords
    IDENTIFIER = auto()
    KEYWORD = auto()
    
    # Operators
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    DOUBLE_SLASH = auto()   # //
    PERCENT = auto()        # %
    CARET = auto()          # ^
    HASH = auto()           # #
    AMPERSAND = auto()      # &
    PIPE = auto()           # |
    TILDE = auto()          # ~
    LT = auto()             # <
    GT = auto()             # >
    LT_EQ = auto()          # <=
    GT_EQ = auto()          # >=
    EQ_EQ = auto()          # ==
    TILDE_EQ = auto()       # ~=
    EQ = auto()             # =
    PLUS_EQ = auto()        # +=
    MINUS_EQ = auto()       # -=
    STAR_EQ = auto()        # *=
    SLASH_EQ = auto()       # /=
    PERCENT_EQ = auto()     # %=
    CARET_EQ = auto()       # ^=
    CONCAT_EQ = auto()      # ..=
    
    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    SEMICOLON = auto()      # ;
    COLON = auto()          # :
    DOUBLE_COLON = auto()   # ::
    COMMA = auto()          # ,
    DOT = auto()            # .
    CONCAT = auto()         # ..
    ARROW = auto()          # ->
    
    # Special
    COMMENT = auto()
    NEWLINE = auto()
    WHITESPACE = auto()
    EOF = auto()
    UNKNOWN = auto()

# Luau keywords
KEYWORDS = {
    'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
    'function', 'if', 'in', 'local', 'nil', 'not', 'or', 'repeat',
    'return', 'then', 'true', 'until', 'while',
    # Luau specific
    'continue', 'type', 'export', 'typeof'
}

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, L{self.line}:C{self.column})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset: int = 1) -> Optional[str]:
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> Optional[str]:
        char = self.current_char()
        if char is None:
            return None
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def make_token(self, token_type: TokenType, value: str, line: int = None, col: int = None) -> Token:
        return Token(
            type=token_type,
            value=value,
            line=line or self.line,
            column=col or self.column
        )
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        """Skip single-line and multi-line comments"""
        if self.current_char() == '-' and self.peek() == '-':
            self.advance()  # -
            self.advance()  # -
            
            # Check for multi-line comment --[[ ]]
            if self.current_char() == '[':
                level = 0
                if self.peek() == '[' or self.peek() == '=':
                    # Count equals signs for long bracket level
                    start_pos = self.pos
                    self.advance()  # [
                    while self.current_char() == '=':
                        level += 1
                        self.advance()
                    if self.current_char() == '[':
                        self.advance()  # [
                        # Find matching close bracket
                        close_pattern = ']' + ('=' * level) + ']'
                        while self.pos < len(self.source):
                            if self.source[self.pos:self.pos + len(close_pattern)] == close_pattern:
                                self.pos += len(close_pattern)
                                return
                            if self.current_char() == '\n':
                                self.line += 1
                                self.column = 1
                            self.advance()
                        return
                    else:
                        self.pos = start_pos
            
            # Single line comment
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_string(self) -> Token:
        """Read string literal (single, double, or long bracket)"""
        start_line = self.line
        start_col = self.column
        quote = self.current_char()
        
        # Long bracket string [[ ]] or [=[ ]=]
        if quote == '[':
            level = 0
            self.advance()  # [
            while self.current_char() == '=':
                level += 1
                self.advance()
            if self.current_char() != '[':
                return self.make_token(TokenType.LBRACKET, '[', start_line, start_col)
            self.advance()  # [
            
            result = []
            close_pattern = ']' + ('=' * level) + ']'
            while self.pos < len(self.source):
                if self.source[self.pos:self.pos + len(close_pattern)] == close_pattern:
                    self.pos += len(close_pattern)
                    self.column += len(close_pattern)
                    return self.make_token(TokenType.STRING, ''.join(result), start_line, start_col)
                if self.current_char() == '\n':
                    self.line += 1
                    self.column = 1
                result.append(self.advance())
            raise SyntaxError(f"Unterminated long string at line {start_line}")
        
        # Regular string
        self.advance()  # opening quote
        result = []
        while self.current_char() and self.current_char() != quote:
            if self.current_char() == '\\':
                self.advance()
                escape_char = self.current_char()
                if escape_char == 'n':
                    result.append('\n')
                elif escape_char == 't':
                    result.append('\t')
                elif escape_char == 'r':
                    result.append('\r')
                elif escape_char == '\\':
                    result.append('\\')
                elif escape_char == '"':
                    result.append('"')
                elif escape_char == "'":
                    result.append("'")
                elif escape_char == '0':
                    result.append('\0')
                elif escape_char and escape_char.isdigit():
                    # Numeric escape \ddd
                    num_str = escape_char
                    self.advance()
                    for _ in range(2):
                        if self.current_char() and self.current_char().isdigit():
                            num_str += self.current_char()
                            self.advance()
                        else:
                            break
                    result.append(chr(int(num_str)))
                    continue
                else:
                    result.append(escape_char or '')
                self.advance()
            elif self.current_char() == '\n':
                raise SyntaxError(f"Unterminated string at line {start_line}")
            else:
                result.append(self.advance())
        
        if self.current_char() != quote:
            raise SyntaxError(f"Unterminated string at line {start_line}")
        self.advance()  # closing quote
        
        return self.make_token(TokenType.STRING, ''.join(result), start_line, start_col)
    
    def read_number(self) -> Token:
        """Read numeric literal"""
        start_line = self.line
        start_col = self.column
        result = []
        
        # Hex number
        if self.current_char() == '0' and self.peek() in ('x', 'X'):
            result.append(self.advance())  # 0
            result.append(self.advance())  # x
            while self.current_char() and (self.current_char().isdigit() or 
                  self.current_char() in 'abcdefABCDEF_'):
                if self.current_char() != '_':
                    result.append(self.advance())
                else:
                    self.advance()
            return self.make_token(TokenType.NUMBER, ''.join(result), start_line, start_col)
        
        # Binary number (Luau)
        if self.current_char() == '0' and self.peek() in ('b', 'B'):
            result.append(self.advance())  # 0
            result.append(self.advance())  # b
            while self.current_char() and self.current_char() in '01_':
                if self.current_char() != '_':
                    result.append(self.advance())
                else:
                    self.advance()
            return self.make_token(TokenType.NUMBER, ''.join(result), start_line, start_col)
        
        # Decimal number
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '_'):
            if self.current_char() != '_':
                result.append(self.advance())
            else:
                self.advance()
        
        # Decimal point
        if self.current_char() == '.' and self.peek() and self.peek().isdigit():
            result.append(self.advance())
            while self.current_char() and (self.current_char().isdigit() or self.current_char() == '_'):
                if self.current_char() != '_':
                    result.append(self.advance())
                else:
                    self.advance()
        
        # Exponent
        if self.current_char() and self.current_char() in ('e', 'E'):
            result.append(self.advance())
            if self.current_char() and self.current_char() in ('+', '-'):
                result.append(self.advance())
            while self.current_char() and self.current_char().isdigit():
                result.append(self.advance())
        
        return self.make_token(TokenType.NUMBER, ''.join(result), start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_line = self.line
        start_col = self.column
        result = []
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            result.append(self.advance())
        
        value = ''.join(result)
        
        if value == 'true' or value == 'false':
            return self.make_token(TokenType.BOOLEAN, value, start_line, start_col)
        elif value == 'nil':
            return self.make_token(TokenType.NIL, value, start_line, start_col)
        elif value in KEYWORDS:
            return self.make_token(TokenType.KEYWORD, value, start_line, start_col)
        else:
            return self.make_token(TokenType.IDENTIFIER, value, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code"""
        self.tokens = []
        
        while self.current_char() is not None:
            start_line = self.line
            start_col = self.column
            char = self.current_char()
            
            # Whitespace
            if char in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Newline
            if char == '\n':
                self.advance()
                continue
            
            # Comments
            if char == '-' and self.peek() == '-':
                self.skip_comment()
                continue
            
            # Strings
            if char in '"\'':
                self.tokens.append(self.read_string())
                continue
            
            # Long bracket string
            if char == '[' and (self.peek() == '[' or self.peek() == '='):
                self.tokens.append(self.read_string())
                continue
            
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Multi-char operators
            two_char = char + (self.peek() or '')
            three_char = two_char + (self.peek(2) or '')
            
            if three_char == '...':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(self.make_token(TokenType.VARARG, '...', start_line, start_col))
                continue
            
            if three_char == '..=':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(self.make_token(TokenType.CONCAT_EQ, '..=', start_line, start_col))
                continue
            
            two_char_tokens = {
                '==': TokenType.EQ_EQ,
                '~=': TokenType.TILDE_EQ,
                '<=': TokenType.LT_EQ,
                '>=': TokenType.GT_EQ,
                '..': TokenType.CONCAT,
                '//': TokenType.DOUBLE_SLASH,
                '::': TokenType.DOUBLE_COLON,
                '->': TokenType.ARROW,
                '+=': TokenType.PLUS_EQ,
                '-=': TokenType.MINUS_EQ,
                '*=': TokenType.STAR_EQ,
                '/=': TokenType.SLASH_EQ,
                '%=': TokenType.PERCENT_EQ,
                '^=': TokenType.CARET_EQ,
            }
            
            if two_char in two_char_tokens:
                self.advance()
                self.advance()
                self.tokens.append(self.make_token(two_char_tokens[two_char], two_char, start_line, start_col))
                continue
            
            # Single char tokens
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '%': TokenType.PERCENT,
                '^': TokenType.CARET,
                '#': TokenType.HASH,
                '&': TokenType.AMPERSAND,
                '|': TokenType.PIPE,
                '~': TokenType.TILDE,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '=': TokenType.EQ,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ';': TokenType.SEMICOLON,
                ':': TokenType.COLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
            }
            
            if char in single_char_tokens:
                self.advance()
                self.tokens.append(self.make_token(single_char_tokens[char], char, start_line, start_col))
                continue
            
            # Unknown character
            self.advance()
            self.tokens.append(self.make_token(TokenType.UNKNOWN, char, start_line, start_col))
        
        self.tokens.append(self.make_token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
