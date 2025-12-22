"""
Luau VM Engine - Advanced Polymorphic Virtual Machine
Version 2.0 - Military Grade Protection
"""

import random
import string
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from .parser import *

# ============== Polymorphic Opcode System ==============

class BaseOpCode(IntEnum):
    """Base opcodes - will be remapped for each generation"""
    # Load/Store
    LOADK = 1
    LOADNIL = 2
    LOADBOOL = 3
    MOVE = 4
    GETGLOBAL = 5
    SETGLOBAL = 6
    GETUPVAL = 7
    SETUPVAL = 8
    
    # Table
    NEWTABLE = 10
    GETTABLE = 11
    SETTABLE = 12
    SETLIST = 13
    
    # Arithmetic
    ADD = 20
    SUB = 21
    MUL = 22
    DIV = 23
    MOD = 24
    POW = 25
    UNM = 26
    IDIV = 27
    
    # Bitwise
    BAND = 30
    BOR = 31
    BXOR = 32
    BNOT = 33
    SHL = 34
    SHR = 35
    
    # Comparison
    EQ = 40
    LT = 41
    LE = 42
    NEQ = 43
    GT = 44
    GE = 45
    
    # Logical
    NOT = 50
    AND = 51
    OR = 52
    
    # Control Flow
    JMP = 60
    JMPT = 61
    JMPF = 62
    JMPNIL = 63
    JMPBACK = 64
    
    # Function
    CALL = 70
    TAILCALL = 71
    RETURN = 72
    CLOSURE = 73
    VARARG = 74
    SELF = 75
    
    # String/Misc
    CONCAT = 80
    LEN = 81
    
    # Special
    NOP = 90
    HALT = 255


class OpcodeMapper:
    """Generate random opcode mapping for each compilation"""
    
    def __init__(self):
        self.base_to_custom: Dict[int, int] = {}
        self.custom_to_base: Dict[int, int] = {}
        self._generate_mapping()
    
    def _generate_mapping(self):
        """Create random opcode mapping"""
        # Get all base opcodes
        base_opcodes = [op.value for op in BaseOpCode]
        
        # Generate shuffled custom opcodes (avoid 0 and 255)
        available = list(range(1, 255))
        random.shuffle(available)
        
        custom_opcodes = available[:len(base_opcodes)]
        
        # Create bidirectional mapping
        for base, custom in zip(base_opcodes, custom_opcodes):
            self.base_to_custom[base] = custom
            self.custom_to_base[custom] = base
        
        # Ensure HALT stays recognizable but randomized
        halt_custom = random.randint(200, 254)
        self.base_to_custom[BaseOpCode.HALT] = halt_custom
        self.custom_to_base[halt_custom] = BaseOpCode.HALT
    
    def encode(self, base_opcode: int) -> int:
        """Convert base opcode to custom opcode"""
        return self.base_to_custom.get(base_opcode, base_opcode)
    
    def decode(self, custom_opcode: int) -> int:
        """Convert custom opcode back to base"""
        return self.custom_to_base.get(custom_opcode, custom_opcode)


# ============== Instruction Encoding ==============

class InstructionEncoder:
    """Multiple instruction encoding formats"""
    
    def __init__(self):
        # Choose random encoding format
        self.format_type = random.randint(0, 3)
        self.xor_key = random.randint(1, 0xFFFFFFFF)
        self.shuffle_map = self._generate_shuffle()
    
    def _generate_shuffle(self) -> List[int]:
        """Generate byte shuffle pattern"""
        pattern = [0, 1, 2, 3]
        random.shuffle(pattern)
        return pattern
    
    def encode_instruction(self, opcode: int, a: int, b: int, c: int) -> int:
        """Encode instruction with selected format"""
        if self.format_type == 0:
            return self._encode_standard(opcode, a, b, c)
        elif self.format_type == 1:
            return self._encode_shuffled(opcode, a, b, c)
        elif self.format_type == 2:
            return self._encode_xor(opcode, a, b, c)
        else:
            return self._encode_complex(opcode, a, b, c)
    
    def _encode_standard(self, op: int, a: int, b: int, c: int) -> int:
        """Standard format: [op:8][a:8][b:8][c:8]"""
        return (op << 24) | (a << 16) | (b << 8) | c
    
    def _encode_shuffled(self, op: int, a: int, b: int, c: int) -> int:
        """Shuffled byte order"""
        bytes_arr = [op, a, b, c]
        shuffled = [bytes_arr[self.shuffle_map[i]] for i in range(4)]
        return (shuffled[0] << 24) | (shuffled[1] << 16) | (shuffled[2] << 8) | shuffled[3]
    
    def _encode_xor(self, op: int, a: int, b: int, c: int) -> int:
        """XOR encoded"""
        base = (op << 24) | (a << 16) | (b << 8) | c
        return base ^ self.xor_key
    
    def _encode_complex(self, op: int, a: int, b: int, c: int) -> int:
        """Complex encoding with rotation and XOR"""
        base = (op << 24) | (a << 16) | (b << 8) | c
        # Rotate left by random amount
        rotated = ((base << 7) | (base >> 25)) & 0xFFFFFFFF
        return rotated ^ self.xor_key
    
    def get_decoder_code(self) -> str:
        """Generate Lua decoder for this encoding"""
        if self.format_type == 0:
            return self._decoder_standard()
        elif self.format_type == 1:
            return self._decoder_shuffled()
        elif self.format_type == 2:
            return self._decoder_xor()
        else:
            return self._decoder_complex()
    
    def _decoder_standard(self) -> str:
        return '''
local function decode(instr)
    local op = bit32.rshift(instr, 24)
    local a = bit32.band(bit32.rshift(instr, 16), 0xFF)
    local b = bit32.band(bit32.rshift(instr, 8), 0xFF)
    local c = bit32.band(instr, 0xFF)
    return op, a, b, c
end'''
    
    def _decoder_shuffled(self) -> str:
        # Reverse shuffle
        reverse_map = [0] * 4
        for i, j in enumerate(self.shuffle_map):
            reverse_map[j] = i
        
        return f'''
local function decode(instr)
    local b0 = bit32.band(bit32.rshift(instr, 24), 0xFF)
    local b1 = bit32.band(bit32.rshift(instr, 16), 0xFF)
    local b2 = bit32.band(bit32.rshift(instr, 8), 0xFF)
    local b3 = bit32.band(instr, 0xFF)
    local bytes = {{b0, b1, b2, b3}}
    local map = {{{reverse_map[0]+1},{reverse_map[1]+1},{reverse_map[2]+1},{reverse_map[3]+1}}}
    return bytes[map[1]], bytes[map[2]], bytes[map[3]], bytes[map[4]]
end'''
    
    def _decoder_xor(self) -> str:
        return f'''
local function decode(instr)
    instr = bit32.bxor(instr, {self.xor_key})
    local op = bit32.rshift(instr, 24)
    local a = bit32.band(bit32.rshift(instr, 16), 0xFF)
    local b = bit32.band(bit32.rshift(instr, 8), 0xFF)
    local c = bit32.band(instr, 0xFF)
    return op, a, b, c
end'''
    
    def _decoder_complex(self) -> str:
        return f'''
local function decode(instr)
    instr = bit32.bxor(instr, {self.xor_key})
    instr = bit32.bor(bit32.rshift(instr, 7), bit32.lshift(bit32.band(instr, 0x7F), 25))
    local op = bit32.rshift(instr, 24)
    local a = bit32.band(bit32.rshift(instr, 16), 0xFF)
    local b = bit32.band(bit32.rshift(instr, 8), 0xFF)
    local c = bit32.band(instr, 0xFF)
    return op, a, b, c
end'''


# ============== Bytecode Compiler ==============

@dataclass
class CompiledChunk:
    """Compiled bytecode chunk"""
    instructions: List[int] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    upvalue_count: int = 0
    param_count: int = 0
    is_vararg: bool = False
    max_stack: int = 256


class AdvancedBytecodeCompiler:
    """Compile AST to polymorphic bytecode"""
    
    def __init__(self):
        self.opcode_mapper = OpcodeMapper()
        self.encoder = InstructionEncoder()
        
        self.instructions: List[int] = []
        self.constants: List[Any] = []
        self.constant_map: Dict[str, int] = {}
        
        self.register_top = 0
        self.max_registers = 0
        
        self.labels: Dict[str, int] = {}
        self.pending_jumps: List[Tuple[int, str]] = []
        self.label_counter = 0
        
        # Scope tracking
        self.locals: Dict[str, int] = {}  # name -> register
        self.scope_stack: List[Dict[str, int]] = []
        
        # Loop tracking for break
        self.loop_stack: List[str] = []  # end labels
    
    def compile(self, ast: Program) -> CompiledChunk:
        """Compile entire program"""
        self._compile_block(ast.body)
        self._emit(BaseOpCode.HALT)
        self._resolve_jumps()
        
        return CompiledChunk(
            instructions=self.instructions,
            constants=self.constants,
            max_stack=self.max_registers + 10
        )
    
    def _emit(self, opcode: int, a: int = 0, b: int = 0, c: int = 0) -> int:
        """Emit encoded instruction"""
        custom_op = self.opcode_mapper.encode(opcode)
        encoded = self.encoder.encode_instruction(custom_op, a, b, c)
        self.instructions.append(encoded)
        return len(self.instructions) - 1
    
    def _emit_jump(self, opcode: int, label: str, a: int = 0) -> int:
        """Emit jump with pending label"""
        idx = self._emit(opcode, a, 0, 0)
        self.pending_jumps.append((idx, label))
        return idx
    
    def _create_label(self) -> str:
        """Create unique label"""
        self.label_counter += 1
        return f"__L{self.label_counter}"
    
    def _mark_label(self, label: str):
        """Mark current position"""
        self.labels[label] = len(self.instructions)
    
    def _resolve_jumps(self):
        """Resolve all pending jumps"""
        for idx, label in self.pending_jumps:
            if label not in self.labels:
                continue
            
            target = self.labels[label]
            offset = target - idx - 1
            
            # Re-encode instruction with offset
            instr = self.instructions[idx]
            
            # Decode, modify, re-encode
            if self.encoder.format_type == 2:
                instr ^= self.encoder.xor_key
            elif self.encoder.format_type == 3:
                instr ^= self.encoder.xor_key
                instr = ((instr >> 7) | (instr << 25)) & 0xFFFFFFFF
            
            op = (instr >> 24) & 0xFF
            a = (instr >> 16) & 0xFF
            
            # Store offset in b,c (signed 16-bit)
            b = (offset >> 8) & 0xFF
            c = offset & 0xFF
            
            self.instructions[idx] = self.encoder.encode_instruction(op, a, b, c)
    
    def _alloc_register(self) -> int:
        """Allocate register"""
        reg = self.register_top
        self.register_top += 1
        self.max_registers = max(self.max_registers, self.register_top)
        return reg
    
    def _free_register(self, count: int = 1):
        """Free registers"""
        self.register_top = max(0, self.register_top - count)
    
    def _push_scope(self):
        """Push new scope"""
        self.scope_stack.append(self.locals.copy())
    
    def _pop_scope(self):
        """Pop scope"""
        if self.scope_stack:
            self.locals = self.scope_stack.pop()
    
    def _add_constant(self, value: Any) -> int:
        """Add constant to pool"""
        # Create unique key
        if isinstance(value, str):
            key = f"s:{value}"
        elif isinstance(value, (int, float)):
            key = f"n:{value}"
        elif isinstance(value, bool):
            key = f"b:{value}"
        elif value is None:
            key = "nil"
        else:
            key = f"o:{id(value)}"
        
        if key in self.constant_map:
            return self.constant_map[key]
        
        idx = len(self.constants)
        self.constants.append(value)
        self.constant_map[key] = idx
        return idx
    
    def _compile_block(self, statements: List[ASTNode]):
        """Compile block of statements"""
        for stmt in statements:
            self._compile_statement(stmt)
    
    def _compile_statement(self, node: ASTNode):
        """Compile single statement"""
        if isinstance(node, LocalStatement):
            self._compile_local(node)
        elif isinstance(node, AssignmentStatement):
            self._compile_assignment(node)
        elif isinstance(node, IfStatement):
            self._compile_if(node)
        elif isinstance(node, WhileStatement):
            self._compile_while(node)
        elif isinstance(node, RepeatStatement):
            self._compile_repeat(node)
        elif isinstance(node, ForNumericStatement):
            self._compile_for_numeric(node)
        elif isinstance(node, ForGenericStatement):
            self._compile_for_generic(node)
        elif isinstance(node, ReturnStatement):
            self._compile_return(node)
        elif isinstance(node, BreakStatement):
            self._compile_break()
        elif isinstance(node, ContinueStatement):
            self._compile_continue()
        elif isinstance(node, DoStatement):
            self._push_scope()
            self._compile_block(node.body.statements)
            self._pop_scope()
        elif isinstance(node, CallStatement):
            reg = self._compile_expression(node.expression)
            self._free_register()
        elif isinstance(node, FunctionDeclaration):
            self._compile_function_decl(node)
    
    def _compile_local(self, node: LocalStatement):
        """Compile local declaration"""
        for i, name in enumerate(node.names):
            reg = self._alloc_register()
            self.locals[name.name] = reg
            
            if i < len(node.values):
                value_reg = self._compile_expression(node.values[i])
                if value_reg != reg:
                    self._emit(BaseOpCode.MOVE, reg, value_reg)
                    self._free_register()
            else:
                self._emit(BaseOpCode.LOADNIL, reg)
    
    def _compile_assignment(self, node: AssignmentStatement):
        """Compile assignment"""
        # Compile all values first
        value_regs = []
        for value in node.values:
            value_regs.append(self._compile_expression(value))
        
        # Assign to targets
        for i, target in enumerate(node.targets):
            value_reg = value_regs[i] if i < len(value_regs) else None
            
            if value_reg is None:
                value_reg = self._alloc_register()
                self._emit(BaseOpCode.LOADNIL, value_reg)
            
            if isinstance(target, Identifier):
                if target.name in self.locals:
                    # Local variable
                    local_reg = self.locals[target.name]
                    self._emit(BaseOpCode.MOVE, local_reg, value_reg)
                else:
                    # Global variable
                    name_idx = self._add_constant(target.name)
                    self._emit(BaseOpCode.SETGLOBAL, value_reg, name_idx)
            
            elif isinstance(target, (IndexExpression, MemberExpression)):
                obj_reg = self._compile_expression(target.object)
                
                if isinstance(target, MemberExpression):
                    key_idx = self._add_constant(target.property.name)
                    key_reg = self._alloc_register()
                    self._emit(BaseOpCode.LOADK, key_reg, key_idx)
                else:
                    key_reg = self._compile_expression(target.index)
                
                self._emit(BaseOpCode.SETTABLE, obj_reg, key_reg, value_reg)
                self._free_register(2)  # obj, key
        
        # Free value registers
        self._free_register(len(value_regs))
    
    def _compile_if(self, node: IfStatement):
        """Compile if statement"""
        end_label = self._create_label()
        
        # Main condition
        else_label = self._create_label()
        cond_reg = self._compile_expression(node.condition)
        self._emit_jump(BaseOpCode.JMPF, else_label, cond_reg)
        self._free_register()
        
        # Then block
        self._push_scope()
        self._compile_block(node.then_block.statements)
        self._pop_scope()
        self._emit_jump(BaseOpCode.JMP, end_label)
        
        # Elseif clauses
        for clause in node.elseif_clauses:
            self._mark_label(else_label)
            else_label = self._create_label()
            
            cond_reg = self._compile_expression(clause.condition)
            self._emit_jump(BaseOpCode.JMPF, else_label, cond_reg)
            self._free_register()
            
            self._push_scope()
            self._compile_block(clause.block.statements)
            self._pop_scope()
            self._emit_jump(BaseOpCode.JMP, end_label)
        
        # Else block
        self._mark_label(else_label)
        if node.else_block:
            self._push_scope()
            self._compile_block(node.else_block.statements)
            self._pop_scope()
        
        self._mark_label(end_label)
    
    def _compile_while(self, node: WhileStatement):
        """Compile while loop"""
        start_label = self._create_label()
        end_label = self._create_label()
        
        self.loop_stack.append(end_label)
        
        self._mark_label(start_label)
        
        cond_reg = self._compile_expression(node.condition)
        self._emit_jump(BaseOpCode.JMPF, end_label, cond_reg)
        self._free_register()
        
        self._push_scope()
        self._compile_block(node.body.statements)
        self._pop_scope()
        
        self._emit_jump(BaseOpCode.JMPBACK, start_label)
        
        self._mark_label(end_label)
        self.loop_stack.pop()
    
    def _compile_repeat(self, node: RepeatStatement):
        """Compile repeat-until loop"""
        start_label = self._create_label()
        end_label = self._create_label()
        
        self.loop_stack.append(end_label)
        
        self._mark_label(start_label)
        
        self._push_scope()
        self._compile_block(node.body.statements)
        
        cond_reg = self._compile_expression(node.condition)
        self._emit_jump(BaseOpCode.JMPF, start_label, cond_reg)
        self._free_register()
        self._pop_scope()
        
        self._mark_label(end_label)
        self.loop_stack.pop()
    
    def _compile_for_numeric(self, node: ForNumericStatement):
        """Compile numeric for loop"""
        start_label = self._create_label()
        end_label = self._create_label()
        
        self.loop_stack.append(end_label)
        self._push_scope()
        
        # Initialize loop variable
        var_reg = self._alloc_register()
        self.locals[node.variable.name] = var_reg
        
        start_reg = self._compile_expression(node.start)
        self._emit(BaseOpCode.MOVE, var_reg, start_reg)
        self._free_register()
        
        # Store end value
        end_reg = self._alloc_register()
        end_val = self._compile_expression(node.end)
        self._emit(BaseOpCode.MOVE, end_reg, end_val)
        self._free_register()
        
        # Store step value
        step_reg = self._alloc_register()
        if node.step:
            step_val = self._compile_expression(node.step)
            self._emit(BaseOpCode.MOVE, step_reg, step_val)
            self._free_register()
        else:
            one_idx = self._add_constant(1)
            self._emit(BaseOpCode.LOADK, step_reg, one_idx)
        
        self._mark_label(start_label)
        
        # Check condition: var <= end
        cond_reg = self._alloc_register()
        self._emit(BaseOpCode.LE, cond_reg, var_reg, end_reg)
        self._emit_jump(BaseOpCode.JMPF, end_label, cond_reg)
        self._free_register()  # cond
        
        # Body
        self._compile_block(node.body.statements)
        
        # Increment: var = var + step
        self._emit(BaseOpCode.ADD, var_reg, var_reg, step_reg)
        
        self._emit_jump(BaseOpCode.JMPBACK, start_label)
        
        self._mark_label(end_label)
        
        self._free_register(3)  # var, end, step
        self._pop_scope()
        self.loop_stack.pop()
    
    def _compile_for_generic(self, node: ForGenericStatement):
        """Compile generic for loop"""
        # Simplified implementation
        self._push_scope()
        
        # Allocate variables
        for var in node.variables:
            reg = self._alloc_register()
            self.locals[var.name] = reg
        
        # Compile body
        self._compile_block(node.body.statements)
        
        self._free_register(len(node.variables))
        self._pop_scope()
    
    def _compile_return(self, node: ReturnStatement):
        """Compile return statement"""
        if node.values:
            for i, value in enumerate(node.values):
                reg = self._compile_expression(value)
                self._emit(BaseOpCode.RETURN, reg, i, len(node.values))
                self._free_register()
        else:
            self._emit(BaseOpCode.RETURN, 0, 0, 0)
    
    def _compile_break(self):
        """Compile break statement"""
        if self.loop_stack:
            self._emit_jump(BaseOpCode.JMP, self.loop_stack[-1])
    
    def _compile_continue(self):
        """Compile continue statement"""
        # Would need start label tracking
        pass
    
    def _compile_function_decl(self, node: FunctionDeclaration):
        """Compile function declaration"""
        # Create closure instruction
        pass
    
    def _compile_expression(self, node: ASTNode) -> int:
        """Compile expression, return register with result"""
        if isinstance(node, NumberLiteral):
            reg = self._alloc_register()
            const_idx = self._add_constant(node.value)
            self._emit(BaseOpCode.LOADK, reg, const_idx)
            return reg
        
        elif isinstance(node, StringLiteral):
            reg = self._alloc_register()
            const_idx = self._add_constant(node.value)
            self._emit(BaseOpCode.LOADK, reg, const_idx)
            return reg
        
        elif isinstance(node, BooleanLiteral):
            reg = self._alloc_register()
            self._emit(BaseOpCode.LOADBOOL, reg, 1 if node.value else 0)
            return reg
        
        elif isinstance(node, NilLiteral):
            reg = self._alloc_register()
            self._emit(BaseOpCode.LOADNIL, reg)
            return reg
        
        elif isinstance(node, Identifier):
            if node.name in self.locals:
                return self.locals[node.name]
            else:
                reg = self._alloc_register()
                name_idx = self._add_constant(node.name)
                self._emit(BaseOpCode.GETGLOBAL, reg, name_idx)
                return reg
        
        elif isinstance(node, BinaryExpression):
            return self._compile_binary(node)
        
        elif isinstance(node, UnaryExpression):
            return self._compile_unary(node)
        
        elif isinstance(node, TableConstructor):
            return self._compile_table(node)
        
        elif isinstance(node, CallExpression):
            return self._compile_call(node)
        
        elif isinstance(node, MethodCallExpression):
            return self._compile_method_call(node)
        
        elif isinstance(node, MemberExpression):
            return self._compile_member(node)
        
        elif isinstance(node, IndexExpression):
            return self._compile_index(node)
        
        elif isinstance(node, FunctionExpression):
            return self._compile_function_expr(node)
        
        else:
            reg = self._alloc_register()
            self._emit(BaseOpCode.LOADNIL, reg)
            return reg
    
    def _compile_binary(self, node: BinaryExpression) -> int:
        """Compile binary expression"""
        left_reg = self._compile_expression(node.left)
        right_reg = self._compile_expression(node.right)
        result_reg = self._alloc_register()
        
        op_map = {
            '+': BaseOpCode.ADD,
            '-': BaseOpCode.SUB,
            '*': BaseOpCode.MUL,
            '/': BaseOpCode.DIV,
            '//': BaseOpCode.IDIV,
            '%': BaseOpCode.MOD,
            '^': BaseOpCode.POW,
            '..': BaseOpCode.CONCAT,
            '==': BaseOpCode.EQ,
            '~=': BaseOpCode.NEQ,
            '<': BaseOpCode.LT,
            '<=': BaseOpCode.LE,
            '>': BaseOpCode.GT,
            '>=': BaseOpCode.GE,
            'and': BaseOpCode.AND,
            'or': BaseOpCode.OR,
        }
        
        opcode = op_map.get(node.operator, BaseOpCode.ADD)
        self._emit(opcode, result_reg, left_reg, right_reg)
        
        self._free_register(2)  # left, right
        return result_reg
    
    def _compile_unary(self, node: UnaryExpression) -> int:
        """Compile unary expression"""
        operand_reg = self._compile_expression(node.operand)
        result_reg = self._alloc_register()
        
        if node.operator == '-':
            self._emit(BaseOpCode.UNM, result_reg, operand_reg)
        elif node.operator == 'not':
            self._emit(BaseOpCode.NOT, result_reg, operand_reg)
        elif node.operator == '#':
            self._emit(BaseOpCode.LEN, result_reg, operand_reg)
        elif node.operator == '~':
            self._emit(BaseOpCode.BNOT, result_reg, operand_reg)
        
        self._free_register()  # operand
        return result_reg
    
    def _compile_table(self, node: TableConstructor) -> int:
        """Compile table constructor"""
        table_reg = self._alloc_register()
        self._emit(BaseOpCode.NEWTABLE, table_reg, len(node.fields))
        
        array_idx = 1
        for field in node.fields:
            if field.key:
                if isinstance(field.key, Identifier):
                    key_idx = self._add_constant(field.key.name)
                    key_reg = self._alloc_register()
                    self._emit(BaseOpCode.LOADK, key_reg, key_idx)
                else:
                    key_reg = self._compile_expression(field.key)
            else:
                key_idx = self._add_constant(array_idx)
                key_reg = self._alloc_register()
                self._emit(BaseOpCode.LOADK, key_reg, key_idx)
                array_idx += 1
            
            value_reg = self._compile_expression(field.value)
            self._emit(BaseOpCode.SETTABLE, table_reg, key_reg, value_reg)
            self._free_register(2)  # key, value
        
        return table_reg
    
    def _compile_call(self, node: CallExpression) -> int:
        """Compile function call"""
        func_reg = self._compile_expression(node.callee)
        
        # Compile arguments
        arg_start = self.register_top
        for arg in node.arguments:
            self._compile_expression(arg)
        
        result_reg = self._alloc_register()
        self._emit(BaseOpCode.CALL, result_reg, func_reg, len(node.arguments))
        
        # Free arg registers
        self._free_register(len(node.arguments))
        self._free_register()  # func
        
        return result_reg
    
    def _compile_method_call(self, node: MethodCallExpression) -> int:
        """Compile method call obj:method(args)"""
        obj_reg = self._compile_expression(node.object)
        
        # Get method
        method_idx = self._add_constant(node.method)
        func_reg = self._alloc_register()
        self._emit(BaseOpCode.SELF, func_reg, obj_reg, method_idx)
        
        # Compile arguments
        for arg in node.arguments:
            self._compile_expression(arg)
        
        result_reg = self._alloc_register()
        self._emit(BaseOpCode.CALL, result_reg, func_reg, len(node.arguments) + 1)  # +1 for self
        
        self._free_register(len(node.arguments))
        self._free_register(2)  # func, obj
        
        return result_reg
    
    def _compile_member(self, node: MemberExpression) -> int:
        """Compile member access obj.property"""
        obj_reg = self._compile_expression(node.object)
        
        key_idx = self._add_constant(node.property.name if isinstance(node.property, Identifier) else str(node.property))
        key_reg = self._alloc_register()
        self._emit(BaseOpCode.LOADK, key_reg, key_idx)
        
        result_reg = self._alloc_register()
        self._emit(BaseOpCode.GETTABLE, result_reg, obj_reg, key_reg)
        
        self._free_register(2)  # key, obj
        return result_reg
    
    def _compile_index(self, node: IndexExpression) -> int:
        """Compile index access obj[key]"""
        obj_reg = self._compile_expression(node.object)
        key_reg = self._compile_expression(node.index)
        
        result_reg = self._alloc_register()
        self._emit(BaseOpCode.GETTABLE, result_reg, obj_reg, key_reg)
        
        self._free_register(2)  # key, obj
        return result_reg
    
    def _compile_function_expr(self, node: FunctionExpression) -> int:
        """Compile function expression"""
        # Simplified - would need full closure compilation
        reg = self._alloc_register()
        self._emit(BaseOpCode.LOADNIL, reg)
        return reg


# ============== Polymorphic VM Generator ==============

class VMGenerator:
    """Generate polymorphic VM interpreter"""
    
    def __init__(self, compiler: AdvancedBytecodeCompiler, chunk: CompiledChunk):
        self.compiler = compiler
        self.chunk = chunk
        self.names = self._generate_names()
        self.encryption_key = random.randint(1, 0xFFFFFFFF)
        self.dispatch_type = random.randint(0, 2)  # Different dispatch methods
    
    def _generate_names(self) -> Dict[str, str]:
        """Generate random variable names"""
        def rand_name():
            patterns = [
                lambda: 'l' + ''.join(random.choices('lI1_', k=random.randint(8, 14))),
                lambda: 'O' + ''.join(random.choices('O0o_', k=random.randint(8, 14))),
                lambda: '_' + ''.join(random.choices(string.ascii_letters + '_', k=random.randint(8, 14))),
            ]
            return random.choice(patterns)()
        
        return {
            'bytecode': rand_name(),
            'constants': rand_name(),
            'registers': rand_name(),
            'pc': rand_name(),
            'stack': rand_name(),
            'decode': rand_name(),
            'dispatch': rand_name(),
            'handlers': rand_name(),
            'run': rand_name(),
            'vm': rand_name(),
            'op': rand_name(),
            'a': rand_name(),
            'b': rand_name(),
            'c': rand_name(),
            'instr': rand_name(),
            'key': rand_name(),
            'decrypt': rand_name(),
        }
    
    def generate(self) -> str:
        """Generate complete VM code"""
        # Encrypt bytecode
        encrypted = self._encrypt_bytecode()
        
        # Generate components
        decryptor = self._generate_decryptor()
        decoder = self.compiler.encoder.get_decoder_code()
        handlers = self._generate_handlers()
        dispatcher = self._generate_dispatcher()
        vm_core = self._generate_vm_core()
        bootstrap = self._generate_bootstrap()
        
        # Randomize order of some components
        components = [handlers, dispatcher, vm_core]
        random.shuffle(components)
        
        code = f'''
-- Virtualized Code
local {self.names['bytecode']} = {{{','.join(str(x) for x in encrypted)}}}
local {self.names['constants']} = {self._serialize_constants()}

{decryptor}

{decoder.replace('decode', self.names['decode'])}

{components[0]}

{components[1]}

{components[2]}

{bootstrap}
'''
        return self._obfuscate_vm_code(code)
    
    def _encrypt_bytecode(self) -> List[int]:
        """Multi-layer bytecode encryption"""
        encrypted = []
        key1 = self.encryption_key
        key2 = (self.encryption_key >> 16) | ((self.encryption_key & 0xFFFF) << 16)
        
        for i, instr in enumerate(self.chunk.instructions):
            # Layer 1: XOR with primary key
            enc = instr ^ key1
            # Layer 2: XOR with position-based key
            enc ^= ((i * 0x1337) & 0xFFFFFFFF)
            # Layer 3: XOR with secondary key
            enc ^= key2
            encrypted.append(enc)
        
        return encrypted
    
    def _generate_decryptor(self) -> str:
        """Generate bytecode decryptor"""
        n = self.names
        key1 = self.encryption_key
        key2 = (self.encryption_key >> 16) | ((self.encryption_key & 0xFFFF) << 16)
        
        return f'''
local {n['key']} = {key1}
local {n['decrypt']} = function(bc)
    local result = {{}}
    local k1 = {key1}
    local k2 = {key2}
    for i = 1, #bc do
        local v = bc[i]
        v = bit32.bxor(v, k2)
        v = bit32.bxor(v, bit32.band((i-1) * 0x1337, 0xFFFFFFFF))
        v = bit32.bxor(v, k1)
        result[i] = v
    end
    return result
end
'''
    
    def _generate_handlers(self) -> str:
        """Generate opcode handlers table"""
        n = self.names
        mapper = self.compiler.opcode_mapper
        
        # Build handler table
        handlers_code = f"local {n['handlers']} = {{}}\n"
        
        # Map each opcode to its handler
        handler_defs = {
            BaseOpCode.LOADK: f"{n['registers']}[a] = {n['constants']}[b + 1]",
            BaseOpCode.LOADNIL: f"{n['registers']}[a] = nil",
            BaseOpCode.LOADBOOL: f"{n['registers']}[a] = (b == 1)",
            BaseOpCode.MOVE: f"{n['registers']}[a] = {n['registers']}[b]",
            BaseOpCode.GETGLOBAL: f"{n['registers']}[a] = getfenv()[{n['constants']}[b + 1]]",
            BaseOpCode.SETGLOBAL: f"getfenv()[{n['constants']}[b + 1]] = {n['registers']}[a]",
            BaseOpCode.NEWTABLE: f"{n['registers']}[a] = {{}}",
            BaseOpCode.GETTABLE: f"{n['registers']}[a] = {n['registers']}[b][{n['registers']}[c]]",
            BaseOpCode.SETTABLE: f"{n['registers']}[a][{n['registers']}[b]] = {n['registers']}[c]",
            BaseOpCode.ADD: f"{n['registers']}[a] = {n['registers']}[b] + {n['registers']}[c]",
            BaseOpCode.SUB: f"{n['registers']}[a] = {n['registers']}[b] - {n['registers']}[c]",
            BaseOpCode.MUL: f"{n['registers']}[a] = {n['registers']}[b] * {n['registers']}[c]",
            BaseOpCode.DIV: f"{n['registers']}[a] = {n['registers']}[b] / {n['registers']}[c]",
            BaseOpCode.MOD: f"{n['registers']}[a] = {n['registers']}[b] % {n['registers']}[c]",
            BaseOpCode.POW: f"{n['registers']}[a] = {n['registers']}[b] ^ {n['registers']}[c]",
            BaseOpCode.UNM: f"{n['registers']}[a] = -{n['registers']}[b]",
            BaseOpCode.IDIV: f"{n['registers']}[a] = math.floor({n['registers']}[b] / {n['registers']}[c])",
            BaseOpCode.BAND: f"{n['registers']}[a] = bit32.band({n['registers']}[b], {n['registers']}[c])",
            BaseOpCode.BOR: f"{n['registers']}[a] = bit32.bor({n['registers']}[b], {n['registers']}[c])",
            BaseOpCode.BXOR: f"{n['registers']}[a] = bit32.bxor({n['registers']}[b], {n['registers']}[c])",
            BaseOpCode.BNOT: f"{n['registers']}[a] = bit32.bnot({n['registers']}[b])",
            BaseOpCode.SHL: f"{n['registers']}[a] = bit32.lshift({n['registers']}[b], {n['registers']}[c])",
            BaseOpCode.SHR: f"{n['registers']}[a] = bit32.rshift({n['registers']}[b], {n['registers']}[c])",
            BaseOpCode.EQ: f"{n['registers']}[a] = ({n['registers']}[b] == {n['registers']}[c])",
            BaseOpCode.LT: f"{n['registers']}[a] = ({n['registers']}[b] < {n['registers']}[c])",
            BaseOpCode.LE: f"{n['registers']}[a] = ({n['registers']}[b] <= {n['registers']}[c])",
            BaseOpCode.NEQ: f"{n['registers']}[a] = ({n['registers']}[b] ~= {n['registers']}[c])",
            BaseOpCode.GT: f"{n['registers']}[a] = ({n['registers']}[b] > {n['registers']}[c])",
            BaseOpCode.GE: f"{n['registers']}[a] = ({n['registers']}[b] >= {n['registers']}[c])",
            BaseOpCode.NOT: f"{n['registers']}[a] = not {n['registers']}[b]",
            BaseOpCode.AND: f"{n['registers']}[a] = {n['registers']}[b] and {n['registers']}[c]",
            BaseOpCode.OR: f"{n['registers']}[a] = {n['registers']}[b] or {n['registers']}[c]",
            BaseOpCode.CONCAT: f"{n['registers']}[a] = tostring({n['registers']}[b]) .. tostring({n['registers']}[c])",
            BaseOpCode.LEN: f"{n['registers']}[a] = #{n['registers']}[b]",
            BaseOpCode.CALL: self._generate_call_handler(),
            BaseOpCode.RETURN: f"return {n['registers']}[a]",
            BaseOpCode.SELF: f"{n['registers']}[a] = {n['registers']}[b][{n['constants']}[c + 1]]; {n['registers']}[a + 1] = {n['registers']}[b]",
        }
        
        for base_op, handler in handler_defs.items():
            custom_op = mapper.encode(base_op)
            # Wrap in function
            handlers_code += f"{n['handlers']}[{custom_op}] = function(a, b, c) {handler} end\n"
        
        # Jump handlers need special treatment
        handlers_code += self._generate_jump_handlers()
        
        return handlers_code
    
    def _generate_call_handler(self) -> str:
        """Generate function call handler"""
        n = self.names
        return f'''
local func = {n['registers']}[b]
local args = {{}}
for i = 1, c do
    args[i] = {n['registers']}[b + i]
end
local results = {{func(unpack(args))}}
{n['registers']}[a] = results[1]
'''
    
    def _generate_jump_handlers(self) -> str:
        """Generate jump instruction handlers"""
        n = self.names
        mapper = self.compiler.opcode_mapper
        
        code = ""
        
        # JMP
        jmp_op = mapper.encode(BaseOpCode.JMP)
        code += f'''
{n['handlers']}[{jmp_op}] = function(a, b, c)
    local offset = bit32.lshift(b, 8) + c
    if offset > 32767 then offset = offset - 65536 end
    {n['pc']} = {n['pc']} + offset
end
'''
        
        # JMPT
        jmpt_op = mapper.encode(BaseOpCode.JMPT)
        code += f'''
{n['handlers']}[{jmpt_op}] = function(a, b, c)
    if {n['registers']}[a] then
        local offset = bit32.lshift(b, 8) + c
        if offset > 32767 then offset = offset - 65536 end
        {n['pc']} = {n['pc']} + offset
    end
end
'''
        
        # JMPF
        jmpf_op = mapper.encode(BaseOpCode.JMPF)
        code += f'''
{n['handlers']}[{jmpf_op}] = function(a, b, c)
    if not {n['registers']}[a] then
        local offset = bit32.lshift(b, 8) + c
        if offset > 32767 then offset = offset - 65536 end
        {n['pc']} = {n['pc']} + offset
    end
end
'''
        
        # JMPBACK
        jmpback_op = mapper.encode(BaseOpCode.JMPBACK)
        code += f'''
{n['handlers']}[{jmpback_op}] = function(a, b, c)
    local offset = bit32.lshift(b, 8) + c
    if offset > 32767 then offset = offset - 65536 end
    {n['pc']} = {n['pc']} + offset
end
'''
        
        return code
    
    def _generate_dispatcher(self) -> str:
        """Generate instruction dispatcher"""
        n = self.names
        halt_op = self.compiler.opcode_mapper.encode(BaseOpCode.HALT)
        
        if self.dispatch_type == 0:
            # Table dispatch
            return f'''
local {n['dispatch']} = function()
    local {n['instr']} = {n['bytecode']}[{n['pc']}]
    {n['pc']} = {n['pc']} + 1
    local {n['op']}, {n['a']}, {n['b']}, {n['c']} = {n['decode']}({n['instr']})
    if {n['op']} == {halt_op} then return false end
    local handler = {n['handlers']}[{n['op']}]
    if handler then handler({n['a']}, {n['b']}, {n['c']}) end
    return true
end
'''
        elif self.dispatch_type == 1:
            # Inline dispatch with if-elseif chain (selected handlers)
            return f'''
local {n['dispatch']} = function()
    local {n['instr']} = {n['bytecode']}[{n['pc']}]
    {n['pc']} = {n['pc']} + 1
    local {n['op']}, {n['a']}, {n['b']}, {n['c']} = {n['decode']}({n['instr']})
    if {n['op']} == {halt_op} then
        return false
    else
        local handler = {n['handlers']}[{n['op']}]
        if handler then handler({n['a']}, {n['b']}, {n['c']}) end
    end
    return true
end
'''
        else:
            # Pcall-wrapped dispatch
            return f'''
local {n['dispatch']} = function()
    local {n['instr']} = {n['bytecode']}[{n['pc']}]
    {n['pc']} = {n['pc']} + 1
    local ok, {n['op']}, {n['a']}, {n['b']}, {n['c']} = pcall({n['decode']}, {n['instr']})
    if not ok then return false end
    if {n['op']} == {halt_op} then return false end
    local handler = {n['handlers']}[{n['op']}]
    if handler then
        local success = pcall(handler, {n['a']}, {n['b']}, {n['c']})
    end
    return true
end
'''
    
    def _generate_vm_core(self) -> str:
        """Generate VM core execution loop"""
        n = self.names
        
        return f'''
local {n['vm']} = function()
    {n['bytecode']} = {n['decrypt']}({n['bytecode']})
    {n['registers']} = {{}}
    {n['pc']} = 1
    
    for i = 0, 255 do {n['registers']}[i] = nil end
    
    while {n['pc']} <= #{n['bytecode']} do
        local running = {n['dispatch']}()
        if not running then break end
    end
end
'''
    
    def _generate_bootstrap(self) -> str:
        """Generate bootstrap code"""
        n = self.names
        
        # Randomize bootstrap style
        style = random.randint(0, 2)
        
        if style == 0:
            return f"coroutine.wrap({n['vm']})()"
        elif style == 1:
            return f"spawn(function() {n['vm']}() end)"
        else:
            return f"task.spawn({n['vm']})"
    
    def _serialize_constants(self) -> str:
        """Serialize constants to Lua table"""
        items = []
        for const in self.chunk.constants:
            if const is None:
                items.append('nil')
            elif isinstance(const, bool):
                items.append('true' if const else 'false')
            elif isinstance(const, (int, float)):
                items.append(str(const))
            elif isinstance(const, str):
                escaped = const.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                items.append(f'"{escaped}"')
            else:
                items.append('nil')
        
        return '{' + ','.join(items) + '}'
    
    def _obfuscate_vm_code(self, code: str) -> str:
        """Additional obfuscation on VM code"""
        # Add fake dead code
        fake_vars = []
        for _ in range(random.randint(3, 6)):
            name = ''.join(random.choices('lI1O0_' + string.ascii_letters, k=random.randint(8, 12)))
            value = random.randint(0, 0xFFFF)
            fake_vars.append(f"local {name} = {value}")
        
        # Insert at random positions
        lines = code.split('\n')
        for fake in fake_vars:
            pos = random.randint(2, len(lines) - 1)
            lines.insert(pos, fake)
        
        return '\n'.join(lines)


# ============== Main Compilation Function ==============

def compile_to_vm(ast: Program) -> str:
    """Compile AST to VM-protected code"""
    compiler = AdvancedBytecodeCompiler()
    chunk = compiler.compile(ast)
    
    generator = VMGenerator(compiler, chunk)
    return generator.generate()
