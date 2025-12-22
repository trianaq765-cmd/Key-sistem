# Lua Obfuscator - Advanced Protection Suite

## Overview

This is a comprehensive Lua bytecode obfuscation and protection system written in Python. The project provides multiple layers of security for Lua scripts including bytecode transformation, multi-layer encryption, anti-tampering measures, and custom VM generation. It supports multiple interfaces: CLI, Web API, and Discord bot integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components

The system follows a modular pipeline architecture with distinct components:

1. **Lua Parser (`lua_parser.py`)** - Parses Lua 5.1 bytecode files, handling opcodes, constants, functions, and the complete chunk structure. This is the foundation that all other components depend on.

2. **Bytecode Transformer (`lua_transformer.py`)** - Applies obfuscation transformations including opcode shuffling, string/number encryption, junk code injection, and control flow flattening. Uses `TransformConfig` dataclass for configuration.

3. **VM Generator (`lua_vm_generator.py`)** - Generates custom Lua virtual machines that can execute the transformed bytecode. Each build produces a unique VM with obfuscated names and structure. Uses `VMConfig` dataclass.

4. **Encryption Layer (`lua_encryption.py`)** - Provides multiple encryption algorithms (XOR variants, RC4, Feistel, custom substitution) with support for layered encryption. Uses `EncryptionConfig` and supports `EncryptionAlgorithm` enum.

5. **Anti-Tamper System (`lua_antitamper.py`)** - Generates protection code including debugger detection, timing checks, integrity verification, and various detection actions. Uses `AntiTamperConfig` with `ProtectionType` and `DetectionAction` enums.

### Integration Layer

1. **Config Manager (`config_manager.py`)** - Unified configuration system supporting JSON, YAML, TOML formats. Provides `ObfuscatorConfig` container that combines all component configs.

2. **Pipeline (`pipeline.py`)** - Orchestrates the complete obfuscation process through defined stages: PARSE → ANALYZE → TRANSFORM → VM_GENERATE → ENCRYPT → ANTITAMPER. Returns `PipelineResult` with metrics.

3. **Main Entry Point (`main.py`)** - Handles dependency checking/installation and routes to appropriate interface (CLI, Web API, or setup).

### Interface Layer

1. **CLI (`cli.py`)** - Command-line interface using argparse for automation and scripting.

2. **Web API (`web_api.py`)** - Flask-based REST API with CORS support, file upload handling, rate limiting, and job tracking. Configurable via `APIConfig`.

3. **Discord Bot (`discord_bot.py`)** - Discord.py bot for obfuscation commands with role-based access control and file processing.

### Design Patterns Used

- **Dataclass Configuration** - All components use Python dataclasses for type-safe configuration
- **Enum-based Options** - IntEnum classes define algorithm choices and protection types
- **Abstract Base Classes** - Used for encryption algorithm implementations
- **Pipeline Pattern** - Sequential processing stages with progress callbacks
- **Factory Pattern** - VM and protection code generation

## External Dependencies

### Python Packages

- **Flask** - Web framework for REST API
- **flask-cors** - CORS handling for API
- **pycryptodome** - Cryptographic primitives (auto-installed by main.py)
- **discord.py** - Discord bot framework
- **aiohttp** - Async HTTP for Discord bot
- **PyYAML** - YAML config file support
- **toml** - TOML config file support

### Environment Variables

The application uses environment variables for configuration:

- `SECRET_KEY` - Flask secret key
- `HOST`, `PORT` - API server binding
- `UPLOAD_FOLDER`, `OUTPUT_FOLDER` - File storage paths
- `MAX_UPLOAD_SIZE` - Upload size limit
- `RATE_LIMIT` - Enable/disable rate limiting
- `DISCORD_TOKEN` - Discord bot token
- `BOT_PREFIX` - Discord command prefix
- `ALLOWED_ROLES`, `ALLOWED_USERS` - Discord access control

### File Formats

- Input: Lua bytecode files (`.luac`, `.lua`)
- Config: JSON, YAML, TOML
- Output: Protected Lua scripts with embedded VM