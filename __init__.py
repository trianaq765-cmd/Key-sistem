"""
Lua Obfuscator Package
Complete suite for Lua code obfuscation with multiple techniques
"""

# Import all modules
from .lexer import *
from .parser import *
from .lua_parser import *  # Additional Lua parser
from .transformer import *
from .lua_transformer import *
from .lua_antitamper import *
from .lua_encryption import *  # Encryption module
from .lua_vm_generator import *
from .luraph_style import *
from .real_vm import RealVMObfuscator, ObfuscationResult
from .vm_engine import *
from .pipeline import *

# Main obfuscation methods
def obfuscate_realvm(code: str, seed: int = None) -> dict:
    """Obfuscate using Real VM method"""
    try:
        obf = RealVMObfuscator(seed)
        result = obf.obfuscate(code)
        
        return {
            "success": result.success,
            "code": result.code,
            "error": result.error,
            "stats": {
                "method": "RealVM",
                "original_size": result.original_size,
                "obfuscated_size": result.obfuscated_size,
                "bytecode_size": result.bytecode_size,
                "instruction_count": result.instruction_count,
                "constant_count": result.constant_count,
                "time_ms": result.time_ms
            }
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def obfuscate_luraph(code: str, **options) -> dict:
    """Obfuscate using Luraph style"""
    try:
        from .luraph_style import LuraphObfuscator
        obf = LuraphObfuscator(**options)
        result = obf.obfuscate(code)
        
        return {
            "success": True,
            "code": result,
            "error": None,
            "stats": {"method": "Luraph"}
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def obfuscate_vm(code: str, **options) -> dict:
    """Obfuscate using VM Generator"""
    try:
        from .lua_vm_generator import VMGenerator
        gen = VMGenerator(**options)
        result = gen.generate(code)
        
        return {
            "success": True,
            "code": result,
            "error": None,
            "stats": {"method": "VMGenerator"}
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def obfuscate_antitamper(code: str, **options) -> dict:
    """Add anti-tampering protection"""
    try:
        from .lua_antitamper import AntiTamper
        at = AntiTamper(**options)
        result = at.protect(code)
        
        return {
            "success": True,
            "code": result,
            "error": None,
            "stats": {"method": "AntiTamper"}
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def encrypt_lua(code: str, key: str = None, algorithm: str = "aes", **options) -> dict:
    """
    Encrypt Lua code
    
    Args:
        code: Lua source code
        key: Encryption key (auto-generated if None)
        algorithm: Encryption algorithm ('aes', 'xor', 'rc4', 'custom')
        **options: Additional encryption options
    
    Returns:
        dict with encrypted code and decryption stub
    """
    try:
        from .lua_encryption import LuaEncryptor
        
        encryptor = LuaEncryptor(algorithm=algorithm)
        
        if key is None:
            import secrets
            key = secrets.token_hex(32)
        
        encrypted_code, decrypt_stub = encryptor.encrypt(code, key, **options)
        
        return {
            "success": True,
            "code": decrypt_stub + encrypted_code,
            "encrypted_code": encrypted_code,
            "decrypt_stub": decrypt_stub,
            "key": key,
            "algorithm": algorithm,
            "error": None,
            "stats": {
                "method": "Encryption",
                "algorithm": algorithm,
                "original_size": len(code),
                "encrypted_size": len(encrypted_code)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def obfuscate_pipeline(code: str, methods: list = None, **options) -> dict:
    """
    Obfuscate using pipeline of multiple methods
    
    Args:
        code: Lua source code
        methods: List of methods to apply in order
                ['realvm', 'antitamper', 'luraph', 'vm', 'encrypt']
        **options: Additional options for each method
    
    Returns:
        dict with obfuscation results
    """
    try:
        from .pipeline import ObfuscationPipeline
        
        if methods is None:
            methods = ['realvm', 'antitamper', 'encrypt']  # Default pipeline
        
        pipeline = ObfuscationPipeline(methods, **options)
        result = pipeline.execute(code)
        
        return {
            "success": True,
            "code": result,
            "error": None,
            "stats": {
                "method": "Pipeline",
                "methods_used": methods
            }
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def transform_code(code: str, transformations: list = None) -> dict:
    """
    Apply transformations to Lua code
    
    Args:
        code: Lua source code
        transformations: List of transformations to apply
    """
    try:
        from .lua_transformer import LuaTransformer
        
        transformer = LuaTransformer()
        
        if transformations is None:
            transformations = ['rename_vars', 'encrypt_strings', 'add_junk']
        
        result = transformer.transform(code, transformations)
        
        return {
            "success": True,
            "code": result,
            "error": None,
            "stats": {"transformations": transformations}
        }
    except Exception as e:
        return {
            "success": False,
            "code": "",
            "error": str(e),
            "stats": {}
        }

def parse_lua(code: str, parser_type: str = "default") -> dict:
    """
    Parse Lua code to AST
    
    Args:
        code: Lua source code
        parser_type: Which parser to use ('default', 'advanced')
    """
    try:
        if parser_type == "advanced":
            from .lua_parser import AdvancedLuaParser
            parser = AdvancedLuaParser()
        else:
            from .parser import LuaParser
            parser = LuaParser()
            
        ast = parser.parse(code)
        
        return {
            "success": True,
            "ast": ast,
            "parser_type": parser_type,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "ast": None,
            "error": str(e)
        }

def tokenize_lua(code: str) -> dict:
    """Tokenize Lua code"""
    try:
        from .lexer import LuaLexer
        lexer = LuaLexer()
        tokens = lexer.tokenize(code)
        
        return {
            "success": True,
            "tokens": tokens,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "tokens": [],
            "error": str(e)
        }

def analyze_lua(code: str) -> dict:
    """
    Analyze Lua code for optimization and obfuscation recommendations
    """
    try:
        from .lua_parser import LuaAnalyzer
        
        analyzer = LuaAnalyzer()
        analysis = analyzer.analyze(code)
        
        return {
            "success": True,
            "analysis": {
                "complexity": analysis.get('complexity', 'medium'),
                "functions": analysis.get('functions', []),
                "variables": analysis.get('variables', []),
                "strings": analysis.get('strings', []),
                "recommendations": analysis.get('recommendations', []),
                "vulnerabilities": analysis.get('vulnerabilities', [])
            },
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "analysis": {},
            "error": str(e)
        }

# Main unified obfuscation function
def obfuscate(code: str, method: str = "auto", level: int = 3, **options) -> dict:
    """
    Universal obfuscation function
    
    Args:
        code: Lua source code
        method: Obfuscation method to use
               - 'auto': Automatic best method based on level
               - 'realvm': Real VM obfuscation
               - 'luraph': Luraph style obfuscation
               - 'vm': VM generator
               - 'antitamper': Anti-tampering only
               - 'encrypt': Encryption only
               - 'pipeline': Multiple methods
               - 'transform': Code transformations only
               - 'maximum': Maximum protection (all methods)
        level: Protection level (1-5)
               1 = Basic (fast, low protection)
               2 = Medium (balanced)
               3 = High (good protection)
               4 = Very High (strong protection)
               5 = Maximum (all protections, slow)
        **options: Method-specific options
    
    Returns:
        dict with obfuscation results
    """
    methods_map = {
        'realvm': obfuscate_realvm,
        'luraph': obfuscate_luraph,
        'vm': obfuscate_vm,
        'antitamper': obfuscate_antitamper,
        'encrypt': encrypt_lua,
        'pipeline': obfuscate_pipeline,
        'transform': transform_code
    }
    
    # Auto mode based on level
    if method == 'auto':
        if level == 1:
            return transform_code(code, ['rename_vars'], **options)
        elif level == 2:
            return obfuscate_antitamper(code, **options)
        elif level == 3:
            return obfuscate_pipeline(code, ['transform', 'antitamper'], **options)
        elif level == 4:
            return obfuscate_pipeline(code, ['realvm', 'antitamper'], **options)
        else:  # level 5
            return obfuscate_pipeline(code, ['realvm', 'encrypt', 'antitamper'], **options)
    
    # Maximum protection
    if method == 'maximum':
        return obfuscate_pipeline(
            code, 
            ['transform', 'realvm', 'luraph', 'encrypt', 'antitamper'], 
            **options
        )
    
    if method in methods_map:
        return methods_map[method](code, **options)
    
    return {
        "success": False,
        "code": "",
        "error": f"Unknown method: {method}",
        "stats": {}
    }

# Batch processing
def obfuscate_batch(files: list, method: str = "auto", output_dir: str = None, **options) -> dict:
    """
    Obfuscate multiple Lua files
    
    Args:
        files: List of file paths
        method: Obfuscation method
        output_dir: Output directory (None = same as input)
        **options: Method options
    
    Returns:
        dict with results for each file
    """
    import os
    from pathlib import Path
    
    results = {}
    
    for file_path in files:
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Obfuscate
            result = obfuscate(code, method, **options)
            
            # Save if successful
            if result['success']:
                if output_dir:
                    output_path = Path(output_dir) / Path(file_path).name
                else:
                    output_path = Path(file_path).with_suffix('.obfuscated.lua')
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result['code'])
                
                result['output_path'] = str(output_path)
            
            results[file_path] = result
            
        except Exception as e:
            results[file_path] = {
                "success": False,
                "error": str(e)
            }
    
    return results

# Utility functions
def get_available_methods() -> list:
    """Get list of available obfuscation methods"""
    return [
        'realvm',
        'luraph', 
        'vm',
        'antitamper',
        'encrypt',
        'pipeline',
        'transform',
        'maximum'
    ]

def get_method_info(method: str) -> dict:
    """Get information about a specific method"""
    info = {
        'realvm': {
            'name': 'Real VM Obfuscator',
            'description': 'Virtual machine based obfuscation',
            'strength': 'Very High',
            'speed': 'Medium',
            'reversible': 'Very Difficult'
        },
        'luraph': {
            'name': 'Luraph Style',
            'description': 'Luraph-like obfuscation techniques',
            'strength': 'High',
            'speed': 'Fast',
            'reversible': 'Difficult'
        },
        'vm': {
            'name': 'VM Generator',
            'description': 'Custom VM bytecode generation',
            'strength': 'High',
            'speed': 'Medium',
            'reversible': 'Difficult'
        },
        'antitamper': {
            'name': 'Anti-Tamper',
            'description': 'Anti-tampering and integrity checks',
            'strength': 'Medium',
            'speed': 'Very Fast',
            'reversible': 'Medium'
        },
        'encrypt': {
            'name': 'Encryption',
            'description': 'Code encryption with various algorithms',
            'strength': 'High',
            'speed': 'Fast',
            'reversible': 'Requires Key'
        },
        'pipeline': {
            'name': 'Pipeline',
            'description': 'Combination of multiple methods',
            'strength': 'Very High',
            'speed': 'Slow',
            'reversible': 'Extremely Difficult'
        },
        'transform': {
            'name': 'Code Transformer',
            'description': 'Basic code transformations',
            'strength': 'Low',
            'speed': 'Very Fast',
            'reversible': 'Easy'
        },
        'maximum': {
            'name': 'Maximum Protection',
            'description': 'All available protection methods combined',
            'strength': 'Maximum',
            'speed': 'Very Slow',
            'reversible': 'Nearly Impossible'
        }
    }
    
    return info.get(method, {})

def benchmark(code: str, methods: list = None) -> dict:
    """
    Benchmark different obfuscation methods
    
    Args:
        code: Lua code to test
        methods: List of methods to benchmark (None = all)
    
    Returns:
        dict with benchmark results
    """
    import time
    
    if methods is None:
        methods = get_available_methods()
    
    results = {}
    
    for method in methods:
        if method == 'pipeline' or method == 'maximum':
            continue  # Skip composite methods
        
        start = time.time()
        result = obfuscate(code, method)
        elapsed = (time.time() - start) * 1000  # ms
        
        results[method] = {
            'success': result['success'],
            'time_ms': elapsed,
            'output_size': len(result['code']) if result['success'] else 0,
            'compression_ratio': len(result['code']) / len(code) if result['success'] else 0
        }
    
    return results

def validate_lua(code: str) -> dict:
    """
    Validate Lua code syntax
    
    Returns:
        dict with validation results
    """
    try:
        # Try parsing with both parsers
        result1 = parse_lua(code, 'default')
        result2 = parse_lua(code, 'advanced')
        
        if result1['success'] or result2['success']:
            return {
                "valid": True,
                "error": None,
                "warnings": []
            }
        else:
            return {
                "valid": False,
                "error": result1['error'] or result2['error'],
                "warnings": []
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "warnings": []
        }

# Export all public functions and classes
__all__ = [
    # Main function
    'obfuscate',
    
    # Batch processing
    'obfuscate_batch',
    
    # Specific obfuscation methods
    'obfuscate_realvm',
    'obfuscate_luraph',
    'obfuscate_vm',
    'obfuscate_antitamper',
    'obfuscate_pipeline',
    'encrypt_lua',
    'transform_code',
    
    # Parser and analysis
    'parse_lua',
    'tokenize_lua',
    'analyze_lua',
    'validate_lua',
    
    # Utility functions
    'get_available_methods',
    'get_method_info',
    'benchmark',
    
    # Classes (for advanced usage)
    'RealVMObfuscator',
    'ObfuscationResult'
]

# Package metadata
__version__ = '2.0.0'
__author__ = 'Lua Obfuscator Team'
__description__ = 'Complete Lua obfuscation suite with encryption and multiple protection techniques'

# Configuration presets
PRESETS = {
    'basic': {
        'method': 'transform',
        'transformations': ['rename_vars']
    },
    'standard': {
        'method': 'pipeline',
        'methods': ['transform', 'antitamper']
    },
    'strong': {
        'method': 'pipeline',
        'methods': ['realvm', 'antitamper']
    },
    'maximum': {
        'method': 'pipeline',
        'methods': ['transform', 'realvm', 'luraph', 'encrypt', 'antitamper']
    },
    'fast': {
        'method': 'antitamper'
    },
    'balanced': {
        'method': 'pipeline',
        'methods': ['luraph', 'encrypt']
    }
}

def obfuscate_with_preset(code: str, preset: str = 'standard', **overrides) -> dict:
    """
    Obfuscate using predefined presets
    
    Args:
        code: Lua source code
        preset: Preset name ('basic', 'standard', 'strong', 'maximum', 'fast', 'balanced')
        **overrides: Override preset settings
    
    Returns:
        dict with obfuscation results
    """
    if preset not in PRESETS:
        return {
            "success": False,
            "code": "",
            "error": f"Unknown preset: {preset}",
            "stats": {}
        }
    
    config = PRESETS[preset].copy()
    config.update(overrides)
    
    method = config.pop('method')
    return obfuscate(code, method, **config)
