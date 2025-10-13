# 🚀 Anthropic API Manifold Pipe for Open WebUI

> **Advanced Anthropic Claude integration with multi-tool orchestration, prompt caching, and extended thinking capabilities**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange.svg)](https://www.anthropic.com/)

---

## 📖 Overview

An advanced Anthropic API integration for Open WebUI that enables Claude models to orchestrate complex multi-tool workflows. Handle sophisticated tasks like: *"Grab my Jira Issues, Research something, create a Confluence Summary for Next Meeting and send it to me via Slack!"* – all in a single request with parallel tool calling and iterative refinement.

### 🎯 Key Highlights

- **Multi-Tool Loop Execution**: Call multiple tools iteratively in the same response
- **Parallel Tool Calling**: Execute independent tools simultaneously for performance
- **Prompt Caching**: Automatic caching for system prompts, tools, and messages
- **Extended Thinking**: Toggle Claude's reasoning process visibility
- **Vision Support**: Process images with automatic preprocessing
- **Code Execution**: Sandboxed Python code execution via Anthropic's tool
- **Web Search Integration**: Built-in web search with citation support
- **1M Token Context**: Support for Claude Sonnet 4's extended context window

---

## ✨ Features

### Core Functionality

| Feature | Status | Description |
|---------|--------|-------------|
| **Anthropic Python SDK** | ✅ | Official SDK integration |
| **Model Auto-Discovery** | ✅ | Fetches available Claude models from API |
| **Tool Call Loop** | ✅ | Multiple tools in single response |
| **Streaming Responses** | ✅ | Real-time output streaming |
| **Fine-grained Tool Streaming** | ✅ | Beta streaming for tool use |
| **Comprehensive Error Handling** | ✅ | Robust error recovery |

### Advanced Capabilities

| Feature | Status | Description |
|---------|--------|-------------|
| **Image Processing/Vision** | ✅ | Process and analyze images |
| **Extended Thinking** | ✅ | Controllable via valve and toggle filter |
| **Web Search Tool** | ✅ | With citations and valve control |
| **Code Execution** | ✅ | Sandboxed Python environment |
| **Prompt Caching** | ✅ | System, user messages, and tools |
| **1M Token Context** | ✅ | Extended context for Sonnet 4 |
| **Token Usage Display** | ✅ | Via source/citation events |
| **Citations** | ✅ | Currently for web_search |

---

## 🗺️ Roadmap

- 📌 **PDF Processing** with caching support
- 📌 **Enhanced Citations** for tool use and document uploads
- 📌 **Improved Memory System** and RAG integration
- 📌 **Files API** integration
- 📌 **UserValves API Key** support
- 📌 **MCP Connector** (pending evaluation of mcpo)

---

## � Installation

1. **Download** the pipe files to your Open WebUI instance
2. **Configure** your Anthropic API key in the pipe settings
3. **Enable** desired toggle filters (Thinking, Web Search, Code Execution)
4. **Start** using Claude models with advanced tool orchestration

---

## 🔧 Configuration

### Valves

- **API Key**: Your Anthropic API key
- **Web Search**: Enable/disable web search tool
- **Thinking Mode**: Show Claude's reasoning process
- **Code Execution**: Enable sandboxed Python execution
- **Cache Display**: Show cache usage statistics
- **Extended Context**: Enable 1M token context window

### Toggle Filters

Three companion toggle filters are included:
- `anthropic_pipe_thinking_toggle.py` - Control thinking visibility
- `anthropic_pipe_web_search_toggle.py` - Enable/disable web search per message
- `anthropic_pipe_code_execution_toggle.py` - Control code execution tool

---

## 📝 Changelog

### v0.3.4 (Latest)
- ✨ Added Claude 4.5 Sonnet support
- 🐛 Fixed final_message bug
- ✨ Added Open WebUI token usage compatibility
- 🔒 Added duplicate tool name validation
- 🔒 Prevented private tool names (starting with "_")

### v0.3.3
- 🐛 Fixed tool call error

### v0.3.2
- 📝 Fixed typo and added changelog

### v0.3.1
- 🐛 Fixed message disappearing after errors

### v0.3 (September 2025)
- ✨ Added Vision support with image preprocessing
- ✨ Added Extended Thinking filter
- ✨ Web Search enforcement toggle
- ✨ Anthropic Code Execution Tool with toggle filter
- ✨ Code execution result formatting
- ✨ Fine-grained tool streaming beta
- ✨ Malformed JSON handling
- ⚡ Improved cache control with dynamic Memory/RAG detection
- ⚡ Ephemeral caching for stable blocks
- ⚡ Refined tool_choice precedence
- ✨ 1M context optional beta header for Sonnet 4
- 🔧 System prompt cleanup and debug refinements

### v0.2 (August 2025)
- 🐛 Fixed caching by moving Memories to Messages
- ✨ Cache usage statistics display
- 🐛 Fixed last chunk not showing in frontend
- 🐛 Fixed defective event_emitters
- 🧹 Removed unnecessary requirements
- ✨ Implemented Web Search valves and error handling
- ✨ Cache_Control for System_Prompt, Tools, and Messages
- 🔧 Refactored for readability and new model support

---

## 🤝 Contributing

Bug reports and feature requests are welcome! Feel free to open an issue if you encounter any problems.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for [Open WebUI](https://github.com/open-webui/open-webui)
- Powered by [Anthropic Claude](https://www.anthropic.com/)

---

**Made with ❤️ for the Open WebUI community**