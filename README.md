# 🚀 Anthropic API Manifold Pipe for Open WebUI

> **Advanced Anthropic Claude integration with multi-tool orchestration, prompt caching, and extended thinking capabilities**

---

## 📖 Overview

An advanced Anthropic API integration for Open WebUI that enables Claude models to orchestrate complex multi-tool workflows. Handle sophisticated tasks like *"Grab my Jira Issues for this week and send me a summary on slack!"* – all in a single request with parallel and sequential tool calling. Features tool search for large toolsets, automatic context management, and experimental PDF native upload.

### 🎯 Key Highlights

| Feature | Description |
|---------|-------------|
| 🔄 **Multi-Tool Loop** | Call multiple tools iteratively in the same response |
| ⚡ **Parallel Execution** | Execute independent tools simultaneously for performance |
| 💾 **Prompt Caching** | Automatic caching for system prompts, tools, and messages, compatible with openwebui RAG and Memory System |
| 🧠 **Extended Thinking** | Toggle Claude's reasoning on and off per-conversation |
| 👁️ **Vision Support** | Process images with Claude |
| 💻 **Code Execution** | Sandboxed Python execution via Anthropic tool |
| 🔍 **Web Search** | Built-in search with inline citations |
| 🔍 **Tool Search** | BM25/Regex search for tools with deferred loading |
| 🧹 **Context Editing** | Auto-clear tool results and thinking blocks |
| 📊 **1M Token Context** | Extended context for Claude Sonnet 4 and 4.5 |

---

## ✨ Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| **Anthropic Python SDK** | Uses the official SDK integration for reliability |
| **Model Auto-Discovery** | Automatically fetches available Claude models from API |
| **Tool Call Loop** | Execute multiple tools in a single response cycle |
| **Streaming Responses** | Real-time output streaming with status updates |
| **Fine-grained Tool Streaming** | Beta streaming for progressive tool use display |
| **Comprehensive Error Handling** | Robust error recovery with retry logic |
| **Task Support** | Automatic title, tag, and follow-up generation |

### Advanced Capabilities

| Feature | Description |
|---------|-------------|
| **Image Processing/Vision** | Process and analyze images (JPEG, PNG, GIF, WebP) |
| **PDF Native Upload** | ⚠️ EXPERIMENTAL: Attach PDFs as base64 to first message |
| **Tool Search** | 🔍 BM25/Regex search for tools with deferred loading to handle large toolsets |
| **Context Editing** | 🧹 Automatic clearing of old tool results and thinking blocks to manage context window |
| **Code Execution** | 💻 Sandboxed Python environment with result formatting and error handling |
| **Web Search** | 🌐 Built-in search with inline citations `[1]` and source references |
| **Extended Thinking** | 🧠 Controllable reasoning mode via valve and toggle filter |
| **Effort Levels** | Configurable effort (`low`, `medium`, `high`) for Opus 4.5 |
| **Web Search Tool** | With inline citations `[1]` and source references |
| **Code Execution** | Sandboxed Python environment with result formatting |
| **Prompt Caching** | 4-level cache control for cost optimization |
| **Memory & RAG** | Intelligent cache-preserving extraction from system prompts |
| **1M Token Context** | Extended context for Sonnet 4 and 4.5 (Tier 4 required) |
| **Token Usage Display** | Context window progress bar in status |
| **Citations** | For web search and document/RAG sources |
| **OpenWebUI Compatibility** | Channel and Notes support |
| **Security Guards** | Duplicate tool name validation, private method blocking |
| **Tool Search** | Use Tool Search Tool to save tokens with big tool descriptions |
| **Context Editing** | Remove old Tool Results and Thinking Blocks from history to cleanup context window |


---

## 🗺️ Roadmap

| Status | Feature | Notes |
|--------|---------|-------|
| ✅ | **PDF Processing** | ✅ IMPLEMENTED: Base64 inline in messages instead of RAG extraction |
| ✅ | **Tool Search** | ✅ IMPLEMENTED: BM25/Regex search with deferred tool loading |
| ✅ | **Context Editing** | ✅ IMPLEMENTED: Auto-clear tool results and thinking blocks |
| ✅ | **Code Execution** | ✅ IMPLEMENTED: Sandboxed Python environment |
| ✅ | **Web Search** | ✅ IMPLEMENTED: Built-in search with citations |
| ✅ | **Extended Thinking** | ✅ IMPLEMENTED: Toggle filter for thinking mode |
| 📌 | **Files API** integration | For native file handling with Code Execution |
| 📌 | **UserValves API Key** support | Per-user API keys |
| 📌 | **Claude Skills** | Implement Usage of Skills |
| 📌 | **Programmatic Tool calling** | Toggle for Programmatic Tool calling |
| 📌 | **Claude Memory** | Evaluate if the Claude Memory System can be hooked into Openwebui Memory |

---

## 📦 Installation

### Option 1: Install from OpenWebUI Community (Recommended)

Install directly from the Open WebUI community:

| Component | Link |
|-----------|------|
| **Main Pipe** | [anthropic_pipe](https://openwebui.com/f/podden/anthropic_pipe) |
| **Thinking Toggle** | [anthropic_pipe_thinking_toggle](https://openwebui.com/f/podden/anthropic_pipe_thinking_toggle) |
| **Web Search Toggle** | [anthropic_web_search_toggle](https://openwebui.com/f/podden/anthropic_web_search_toggle) |
| **Code Execution Toggle** | [anthropic_pipe_code_execution_toggle](https://openwebui.com/f/podden/anthropic_pipe_code_execution_toggle) |

### Option 2: Manual Installation

1. Go to **Admin Settings** → **Functions**
2. Click **"+ New Function"**
3. Copy the source code from this repository
4. Set name, ID, and description
5. Repeat for each filter

### Configuration Steps

1. **Configure API Key**: Set your Anthropic API key in the pipe settings

2. **Configure Models** (Admin Settings → Models):
   - Activate the Thinking, Web Search, and Code Execution Filters for each Claude model
   - Set **Function Calling** to `Native` in Advanced Parameters
   - Optional: Deactivate OpenWebUI's built-in WebSearch and Code Interpreter
3. **Start chatting** with Claude models!

---

## 🔧 Configuration

### Valves (Global Settings)

| Valve | Default | Description |
|-------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | Your Anthropic API key (required) |
| `ENABLE_1M_CONTEXT` | `false` | Enable 1M token context window (requires Tier 4 API) |
| `WEB_SEARCH` | `true` | Enable web search tool availability |
| `MAX_TOOL_CALLS` | `15` | Maximum tool execution loops per request (1-50) |
| `MAX_RETRIES` | `3` | Maximum retries for failed requests (0-50) |
| `CACHE_CONTROL` | `disabled` | Prompt caching scope (see below) |
| `WEB_SEARCH_USER_*` | - | Default location for web searches (city, region, country, timezone) |
| `ENABLE_TOOL_SEARCH` | `false` | Enable tool search. Allows Claude to search for tools by name/description when many tools are available. |
| `TOOL_SEARCH_TYPE` | `bm25` | Type of tool search: 'regex' for pattern matching or 'bm25' for natural language search. |
| `TOOL_SEARCH_MAX_DESCRIPTION_LENGTH` | `100` | Maximum tool description length. Tools with longer JSON definitions will be deferred for lazy loading. |
| `TOOL_SEARCH_EXCLUDE_TOOLS` | `["web_search", "code_execution_20250825"]` | Tools to exclude from defer_loading when tool search is enabled. These tools will always be loaded immediately. |
| `CONTEXT_EDITING_STRATEGY` | `none` | Context editing strategy: none (disabled), clear_tool_results, clear_thinking, or clear_both. |
| `CONTEXT_EDITING_THINKING_KEEP` | `5` | How many thinking blocks to keep when clearing thinking. |
| `CONTEXT_EDITING_TOOL_TRIGGER` | `50000` | Token count threshold that triggers tool result clearing. |
| `CONTEXT_EDITING_TOOL_KEEP` | `5` | Number of recent tool results to preserve when clearing. |
| `CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST` | `10000` | Minimum tokens to clear when triggered (helps with cache optimization). |
| `CONTEXT_EDITING_TOOL_CLEAR_TOOL_INPUT` | `false` | Also clear tool input parameters when clearing tool results. |

#### Cache Control Options

| Option | Description |
|--------|-------------|
| `cache disabled` | No caching |
| `cache tools array only` | Cache tool definitions |
| `cache tools array and system prompt` | Cache tools + system prompt |
| `cache tools array, system prompt and messages` | Full caching (recommended) |

> 💡 **RAG & MEMORY**: The Pipe intelligently tries to extract Memories and RAG Content from the System Promt and puts it in the last User Message. While this is drasticly saving money and improving speed there maybe some problems with other filters or the models using the Knowledge or Memories wrong. Please open an issue if you have problems.

### UserValves (Per-User Settings)

| Valve | Default | Range | Description |
|-------|---------|-------|-------------|
| `ENABLE_THINKING` | `false` | - | Enable Extended Thinking mode |
| `THINKING_BUDGET_TOKENS` | `4096` | 0-32000 | Token budget for thinking |
| `EFFORT` | `high` | low/medium/high | Effort level for Opus 4.5 |
| `USE_PDF_NATIVE_UPLOAD` | `false` | - | ⚠️ EXPERIMENTAL: Attach PDFs as base64 instead of RAG |
| `SHOW_TOKEN_COUNT` | `false` | - | Show context window progress bar |
| `WEB_SEARCH_MAX_USES` | `5` | 1-20 | Max searches per request |
| `WEB_SEARCH_USER_*` | - | - | Override global location settings |

### Toggle Filters

| Filter | Purpose |
|--------|---------|
| `anthropic_pipe_thinking_toggle.py` | 🧠 Enable thinking mode for the next message |
| `anthropic_pipe_web_search_toggle.py` | 🔍 Force web search for the next message |
| `anthropic_pipe_code_execution_toggle.py` | 💻 Enable code execution for the next message |

---

## 📝 Changelog

### v0.5.9 (Latest) ⚠️ EXPERIMENTAL
- ✨ Added native PDF upload support via `USE_PDF_NATIVE_UPLOAD` UserValve
  - Attaches entire PDF as base64 to first user message for cache compatibility
  - Anthropic parses PDF directly (charts, images, layouts) instead of RAG extraction
  - Automatically filters duplicate RAG content for PDFs
  - Removes entire RAG prompt if PDF is the only source (token savings)
  - **Status**: Experimental and untested - use with caution

### v0.5.8
- 🐛 Fixed UnboundLocalError for 'total_usage' variable when opening new chats
- ✨ Added code execution to default TOOL_SEARCH_EXCLUDE_TOOLS list

### v0.5.7
- ✨ Added Valve to exclude specific tools from deferred loading when tool search is enabled (web_search excluded by default)
- 🌐 Web Search Toggle Filter overrides WEB_SEARCH Valve
- 🐛 Fixed a Bug in Tool Search return

### v0.5.6
- ✨ Added Context Editing feature (clear_tool_uses, clear_thinking) with configurable strategies
- 🔍 Added Tool Search feature (BM25/Regex) with deferred tool loading
- 📊 Status events for context clearing with token counts
- ⚠️ Warning notification for thinking+cache conflict

### v0.5.4
- 🐛 Fixed Message Caching Problems when using RAG or Memories

### v0.5.3
- ✨ Added Support for Anthropic Effort Levels (`low`, `medium`, `high`)
- ✨ Added Support for Opus 4.5
- 🔧 Use correct logger for logging
- 🗑️ Removed DEBUG Valve
- ✨ Introduced UserValves for user-specific settings

### v0.5.2
- 🐛 Fixed usage statistics accumulation for multi-step tool calls
- ✅ Correctly sums input and output tokens across all turns

### v0.5.1
- 🐛 Fixed caching issue in tool execution loops
- ⚡ Optimized caching for multi-step tool calls

### v0.5.0
- 🔒 **CRITICAL FIX**: Eliminated cross-talk between concurrent users/requests
- 🗑️ Removed shared instance state that caused response mixing

<details>
<summary><b>v0.4.x</b> (click to expand)</summary>

#### v0.4.9
- ⚡ Performance optimization: Moved local imports to top level
- 🐛 Fixed fallback logic for model fetching

#### v0.4.8
- ✨ Added configurable `MAX_TOOL_CALLS` valve (default: 15)
- ⚡ Immediate tool execution status events
- ⚠️ Proactive warnings when approaching tool call limit

#### v0.4.7
- 🔒 Fixed potential data leakage between concurrent users

#### v0.4.6
- ✨ Tool results now display input parameters at the top

#### v0.4.5
- ✨ Added status events for local tool execution
- 🎯 Better UX for long-running tools

#### v0.4.4
- ⚡ Tool calls now execute in parallel
- 🐛 Fixed server tool identification

#### v0.4.3
- 🐛 Fixed compatibility with OpenWebUI "Chat with Notes"

#### v0.4.2
- 🐛 Fixed NoneType error in OpenWebUI Channels

#### v0.4.1
- ✨ Added token count display valve
- ✨ Auto-enable native function calling

#### v0.4.0
- ✨ Added Task Support (titles, tags, follow-ups)
- 🐛 Fixed server + local tool use conflict

</details>

<details>
<summary><b>v0.3.x</b> (click to expand)</summary>

#### v0.3.9
- ✨ Added fine-grained cache control valve (4 levels)

#### v0.3.8
- 🗑️ Removed MAX_OUTPUT_TOKENS valve
- ⚡ Reworked caching with OpenWebUI Memory System
- ✨ Added retry logic for transient errors

#### v0.3.7
- 🐛 Fixed Extended Thinking compatibility with Tool Use

#### v0.3.6
- ✨ Added Claude 4.5 Haiku Model

#### v0.3.5
- 🐛 Fixed last chunk not sent bug
- ✨ Added correct citation handling for Web Search

#### v0.3.4
- ✨ Added Claude 4.5 Sonnet support
- ✨ Added OpenWebUI token usage compatibility
- 🔒 Added duplicate tool name validation

#### v0.3.3 - v0.3.1
- Various bug fixes

#### v0.3.0 (September 2025)
- ✨ Added Vision support
- ✨ Added Extended Thinking filter
- ✨ Added Web Search enforcement toggle
- ✨ Added Anthropic Code Execution Tool
- ⚡ Improved cache control with dynamic Memory/RAG detection
- ✨ Added 1M context beta header for Sonnet 4

</details>

<details>
<summary><b>v0.2.0</b> (August 2025)</summary>

- 🐛 Fixed caching by moving Memories to Messages
- ✨ Cache usage statistics display
- 🐛 Fixed last chunk not showing in frontend
- ✨ Implemented Web Search valves and error handling
- ✨ Added Cache_Control for System_Prompt, Tools, and Messages

</details>

---

## 🤝 Contributing

Bug reports and feature requests are welcome! Feel free to [open an issue](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe/issues) if you encounter any problems.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for [Open WebUI](https://github.com/open-webui/open-webui)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Thanks Balaxxe and nbellochi for their original Pipe
---
