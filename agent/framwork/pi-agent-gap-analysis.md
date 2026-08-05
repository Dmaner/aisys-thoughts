```
Tags: #Agent #Framework #Pi #ClaudeCode #Codex #OpenClaw #Hermes
Desc: 中文版。基于公开源码、本机安装包和官方资料，对原生 Pi 与 Claude Code、Codex、OpenClaw、Hermes Agent 的功能差距做代码级对比。
```

# 原生 Pi 与 Agent Framework 的代码级差距分析

> 范围说明：这里的“Pi”指 Inflection AI / pi.ai 的原生 Pi 助手。如果你指的是另一个内部项目或同名产品，结论需要重新核对。
>
> 分析日期：2026-06-30。Agent 产品变化很快，下文只写本次能从公开源码、本机安装包、官方页面中定位到的事实；没有证据的地方明确写“未见公开证据”。

## 结论先行

原生 Pi 和 Claude Code、Codex、OpenClaw、Hermes Agent 的差距不是“少几个工具按钮”，而是缺少一个可公开验证的 agent runtime。

从公开资料看，Pi 是面向个人对话、语音和情绪支持的 AI 助手。它可以作为聊天/解释/陪伴入口，但没有公开证据显示它具备这些 agent framework 的核心层：工作区文件读写、shell 执行、patch/diff、权限审批、沙箱、MCP/插件、hooks、skills、子代理、常驻 gateway、cron、会话/任务持久化、provider/backend 切换、轨迹生成或代码工作流。

相比之下：

- Claude Code 是闭源二进制产品，但官方文档和本机 npm 包都明确定位为 terminal/IDE/GitHub 里的 agentic coding tool。
- Codex 的 CLI/runtime 有公开 Rust 源码，代码里能看到多代理控制面、线程持久化、apply_patch、request_permissions、MCP runtime、hooks、skills、沙箱/执行相关 crate。
- OpenClaw 的 TypeScript 源码是个人助理 gateway：多渠道入口、session/subagent、cron、node-host 执行、sandbox backend、工具策略、DM guardrail 都能定位到源码。
- Hermes Agent 的 Python 源码更强调“自改进 agent”：工具并发执行、tool loop guardrail、memory manager、skill slash command、cron scheduler、gateway platform registry、MCP client、provider registry、多 terminal backend 都能定位到源码。

## 证据分层

| 层级 | 本文如何使用 |
| --- | --- |
| P0：源码证据 | 本次拉取并阅读了 Codex、OpenClaw、Hermes Agent 的公开仓库源码。此类证据可以支撑“项目实现了某个 runtime 模块”。 |
| P1：安装包/官方文档证据 | Claude Code 本机安装包是二进制分发，无法像开源项目一样审源码；本文只使用其 npm 包 metadata、README 和 Anthropic 官方文档，不推断内部实现。 |
| P2：官方产品页/帮助页证据 | Pi 只按公开产品页和帮助中心处理。本文不声称 Pi 内部没有某能力，只说“未见公开证据”。 |

本次源码快照：

| 项目 | 快照 |
| --- | --- |
| Codex | `openai/codex@cfead68e5d3984b247cf0758e3e53b19165de848` |
| OpenClaw | `openclaw/openclaw@56c2d637d9405d00b38a2ca87b92a25f8622d93d` |
| Hermes Agent | `NousResearch/hermes-agent@14c4a849b7b501ffa2eedcf15b92ab5347418aa0` |
| Claude Code | 本机 npm 包 `@anthropic-ai/claude-code@2.1.123`，入口为 `bin/claude.exe` 二进制 |
| Codex CLI | 本机 npm 包 `@openai/codex@0.139.0` |

## 代码级证据

### Pi：只看到对话产品证据，未看到 agent runtime

Pi 的官方入口和帮助页支持把它理解为个人 AI 助手、聊天和语音产品。本文没有找到公开源码，也没有在官方页面中看到本地代码仓库、shell、git、权限审批、MCP、插件、沙箱、cron 或常驻 gateway 的说明。

因此对 Pi 的判断必须保守：不是“Pi 一定不能做这些”，而是“截至本次核对，公开材料没有显示它是一个可执行代码/工具的 agent framework”。

### Claude Code：闭源 coding agent 产品，不能做源码级断言

本机安装的 `@anthropic-ai/claude-code@2.1.123` 包含 `package.json`、README、`sdk-tools.d.ts` 和 `bin/claude.exe`。`package.json` 的描述明确写到它能在终端中使用 Claude、理解代码库、编辑文件、运行命令并处理工作流；README 也写到它运行在 terminal/IDE/GitHub 里，能执行 routine tasks、解释复杂代码、处理 git workflows。

但这个 npm 包不是可读实现源码，主程序是二进制。因此本文对 Claude Code 只做产品能力对比，不把它当作可审计源码样本。它和 Pi 的确定差距来自官方/安装包事实：Claude Code 被设计为 coding agent；Pi 的公开页面没有展示类似代码执行 runtime。

### Codex：开源 runtime 里能看到完整 coding-agent 控制面

Codex 的公开源码里，agent framework 不是文档口号，而是 Rust crate 和 handler 组成的运行时：

- 多代理控制面：`codex-rs/core/src/agent/control.rs:86-91` 说明 `AgentControl` 是 multi-agent operations 的 control-plane handle，并且 root thread/session tree 共享同一个 registry；同文件还实现 `send_input`、`list_agents`、completion watcher 和 parent/child agent 通信。
- 子代理注册与限额：`codex-rs/core/src/agent/registry.rs:16-26` 说明 `AgentRegistry` 用于限制每个用户 session 的 sub-agent 总数；`reserve_spawn_slot`、`register_root_thread`、`reserve_agent_path` 等逻辑维护 agent tree、nickname、path 和 max_threads。
- patch 执行：`codex-rs/core/src/tools/handlers/apply_patch.rs:594-635` 构造 `ApplyPatchRequest`，通过 `ToolOrchestrator` 和 `ApplyPatchRuntime` 执行并返回 delta。这是“能改文件并把变更作为工具结果追踪”的核心。
- 权限请求：`codex-rs/core/src/tools/handlers/request_permissions.rs:66-103` 解析 requested permissions，归一化 additional permissions，然后调用 session 的 `request_permissions_for_environment`。这说明权限是运行时工具链的一部分，不只是提示词。
- MCP runtime：`codex-rs/core/src/session/mcp.rs:79-102` 根据 config、thread init、extension data 和环境 capability roots 生成 runtime MCP config；`309-360` 会刷新 MCP servers、认证状态和 permission profile。
- hooks：`codex-rs/hooks/src/registry.rs:29-39` 定义 hooks 配置；同文件暴露 `run_session_start`、`run_pre_tool_use`、`run_permission_request`、`run_post_tool_use`、`run_pre_compact`、`run_post_compact` 等生命周期事件。
- skills：`codex-rs/core/src/context/available_skills_instructions.rs:26-44` 从 `AvailableSkills` 生成模型上下文，并注入 “How to use skills” 指令。
- 线程持久化：`codex-rs/thread-store/src/types.rs:67-100` 的 `CreateThreadParams` 持久化 session id、thread id、fork/subagent 来源、dynamic tools、selected capability roots、multi-agent version、history mode 等；`420-471` 的 thread summary 保存 cwd、CLI version、source、agent nickname/path、git_info、approval mode、permission profile、token usage 和 history。
- repo 结构还能看到 `exec`、`exec-server`、`apply-patch`、`linux-sandbox`、`windows-sandbox-rs`、`mcp-server`、`codex-mcp`、`cloud-tasks`、`connectors`、`ext/skills`、`ext/memories` 等 crate。

这些源码共同说明：Codex 是一个围绕代码仓库、工具调用、权限、沙箱、MCP、线程、子代理构建的 agent runtime。Pi 的公开材料没有对应层。

### OpenClaw：个人助理 gateway，不只是聊天 UI

OpenClaw 的 README 和源码对得上：它不是单一聊天框，而是 local-first gateway + 多渠道入口 + host/sandbox 工具执行。

可定位源码点：

- 产品定位：`README.md:21-27` 写它运行在用户设备上，通过 WhatsApp、Telegram、Slack、Discord、Google Chat、Signal、iMessage、Feishu、WeChat、QQ 等渠道回答；`README.md:162-176` 列出 local-first Gateway、multi-channel inbox、multi-agent routing、voice、live canvas、tools、companion apps、skills 和 sandbox security model。
- session/subagent：`src/agents/tools/sessions-spawn-tool.ts:1-5` 说明 `sessions_spawn` 会启动 subagent 或 ACP-backed session，并继承 tool policy 和 delivery context；`162-240` 的 schema 支持 task、runtime、model、cwd、thread、mode、cleanup、sandbox、context、attachments、ACP resume/stream 等参数。
- cron：`src/agents/tools/cron-tool.ts:57-73` 定义 `status/list/get/add/update/remove/run/runs/wake` 等动作，支持 `at/every/cron` schedule、delivery mode、payload kind；`112-165` 的 schema 允许 agentTurn/systemEvent、model override、fallback、toolsAllow。
- 工具策略：`src/agents/tool-policy.ts:21-26` 定义 allow/deny policy；`130-195` 支持 plugin tool group 展开；`198-240` 识别 MCP allowlist entry。
- host 执行审批：`src/node-host/invoke-system-run.ts:1-2` 明确这是 approved `system.run` request 的 policy/execution pipeline；`195-233` 合并 global/agent exec policy、mode policy、approval decision、security/ask；`138-144` 还有 cwd drift、script operand binding/drift 的 deny message。
- sandbox：`src/agents/sandbox.ts:1-6` 把 sandbox config、backend、Docker、SSH、filesystem、policy 作为统一 export surface；`src/agents/sandbox/backend.ts:35-45` 是进程级 backend registry，`111-121` 默认注册 Docker 和 SSH backend。
- sandbox 工具 allow/deny：`src/agents/sandbox/tool-policy.ts:1-5` 说明会合并 global、agent、default allow/deny；`186-215` 判断工具是否被 deny 或不在 allowlist。
- DM 入口 guardrail：`src/channels/direct-dm-guard-policy.ts:8-28` 在解密前限制 event kind、future timestamp skew、ciphertext/plaintext 大小、per-sender/global rate limit。

所以 OpenClaw 与 Pi 的差距集中在“入口和运行时控制面”：Pi 是应用内对话，OpenClaw 是用户自管 gateway，可以把远程消息路由到隔离 agent、工具和沙箱。

### Hermes Agent：自改进 agent，运行时覆盖工具、记忆、技能、cron、provider 和 gateway

Hermes Agent 的 README 声称 self-improving、memory、skills、multi-channel gateway、cron、subagents、多 terminal backend。源码中能看到对应实现：

- 工具执行：`agent/tool_executor.py:1-10` 把 sequential/concurrent tool dispatch 从 agent 主循环中抽出来；`69-71` 设置最大并发 worker；`74-90` 在 tool progress 后 best-effort flush session DB，避免有副作用的工具调用丢失 transcript。
- 工具范围：`agent/tool_executor.py:177-223` 为 Tool Search 缓存当前 session 的 scoped deferrable tool names，防止受限 toolset 的 session 通过 unwrap 调到未授权工具。
- tool loop guardrail：`agent/tool_guardrails.py:20-60` 区分 idempotent 和 mutating tools；`63-81` 定义 warn/block/halt 阈值；`145-162` 的 decision 明确支持 allow/warn/block/halt。
- memory manager：`agent/memory_manager.py:1-24` 是 memory providers 的统一集成点，覆盖 system prompt、pre-turn prefetch、post-turn sync；`49-79` 会规范化 provider tool schema，避免坏 schema 毒化整次请求；`163-168`、`171-260` 会 scrub memory-context，降低上下文泄漏风险。
- skill slash command：`agent/skill_commands.py:1-5` 说明 CLI 和 gateway 都能通过 `/skill-name` 调 skills；`58-83` 从 skill-expanded message 恢复用户真实 instruction，避免 memory provider 存下整个 skill body；`138-203` 支持按名字/路径加载 skill。
- cron scheduler：`cron/scheduler.py:1-8` 说明 gateway 后台线程每 60 秒 tick due jobs，并用文件锁避免重入；`115-135` 强制 cron agent 禁用 `cronjob`、`messaging`、`clarify` 等 toolsets；`169-199` 解析 per-job/platform enabled toolsets；`201-229` 定义投递平台和 home target env vars。
- gateway platform registry：`gateway/platform_registry.py:1-10` 允许 built-in/plugin platform adapters 自注册，gateway 不需要硬编码所有平台；`38-159` 的 `PlatformEntry` 包括 adapter factory、required env、allowed users、message limit、cron delivery、standalone sender 等元数据；`162-183` 支持 deferred loaders，避免启动时加载所有重依赖。
- ACP 权限桥接：`acp_adapter/permissions.py:1-2` 明确是 Hermes dangerous-command approvals 的 ACP bridge；`41-70` 构造 allow once/session/always/deny options；`107-167` 把 ACP permission outcome 映射为 Hermes approval 字符串。
- terminal backend：`tools/terminal_tool.py:3-18` 写明 terminal tool 支持 local、Docker、Modal、SSH、Singularity、Daytona 等环境和 background task；`tools/environments/__init__.py:1-9` 说明所有 backend 共享 `BaseEnvironment` 接口，由 `TERMINAL_ENV` 选择。
- MCP：`tools/mcp_tool.py:3-12` 支持 stdio、HTTP/StreamableHTTP、SSE transport，把外部 MCP server 的 tools 注册进 Hermes tool registry；`54-66` 列出自动重连、env 过滤、credential stripping、timeout、sampling、parallel tool call opt-in。
- provider registry：`hermes_cli/providers.py:1-18` 写明 provider identity 来自 models.dev、Hermes overlays、用户配置三层合并；`46-214` 覆盖 OpenRouter、Nous、OpenAI/Codex、xAI、Qwen、LM Studio、Copilot、Anthropic、DeepSeek、Bedrock 等 provider overlay；`providers/base.py:38-80` 定义 provider profile 的 auth、endpoint、vision、fallback models 等字段。

Hermes 与 Pi 的差距不只是“能写代码”，而是“能把经验转成 memory/skills，并长期运行在不同平台和 backend 上”。这类 operational agent state 在 Pi 的公开材料中没有看到。

## 功能矩阵

图例：`源码证据` 表示本次在公开仓库里定位到实现；`官方/安装包证据` 表示来自官方文档或本机安装包；`未见公开证据` 表示本次没有找到 Pi 的公开证据，不等于内部一定没有。

| 能力 | 原生 Pi | Claude Code | Codex | OpenClaw | Hermes Agent |
| --- | --- | --- | --- | --- | --- |
| 主要定位 | 个人对话/语音助手 | coding agent | coding agent/runtime | 个人助理 gateway | 自改进 agent |
| 公开源码可审 | 未见公开源码 | 未见可读实现源码，npm 二进制 | 源码证据 | 源码证据 | 源码证据 |
| 本地代码仓库读写 | 未见公开证据 | 官方/安装包证据 | 源码证据：patch/thread/workspace | 源码证据：host/sandbox tools | 源码证据：terminal/file/tool registry |
| shell / command 执行 | 未见公开证据 | 官方/安装包证据 | 源码证据：exec/request permissions | 源码证据：node-host system.run | 源码证据：terminal tool/backends |
| 权限审批 | 未见公开证据 | 官方文档证据 | 源码证据：request_permissions、permission_profile | 源码证据：exec policy、allowlist、DM guard | 源码证据：approval callback、ACP bridge |
| 沙箱/隔离 backend | 未见公开证据 | 官方文档证据 | 源码证据：linux/windows sandbox crate | 源码证据：Docker/SSH sandbox backend | 源码证据：local/Docker/SSH/Singularity/Modal/Daytona |
| MCP / connector | 未见公开证据 | 官方文档证据 | 源码证据：MCP runtime | 源码证据：tool policy 识别 MCP entries | 源码证据：stdio/HTTP/SSE MCP client |
| skills / hooks / plugins | 未见公开证据 | 官方文档证据 | 源码证据：skills + hooks + plugins/ext | 源码证据：skills + plugin/tool groups | 源码证据：skills + plugin/provider registries |
| 子代理/并行工作 | 未见公开证据 | 官方文档证据 | 源码证据：AgentControl/AgentRegistry | 源码证据：sessions_spawn/subagent registry | 源码证据：concurrent tool dispatch、delegate/subagent |
| 会话/任务持久化 | 聊天历史层面可能有，未见 agent runtime 证据 | 官方文档证据 | 源码证据：thread-store/history/git/approval metadata | 源码证据：gateway sessions/subagent registry | 源码证据：session DB flush、memory/session search |
| cron / scheduled automation | 未见公开证据 | 官方文档证据 | 产品/源码均有相关 surface | 源码证据：cron tool | 源码证据：cron scheduler |
| 多渠道 gateway | Pi web/app/voice | 官方集成证据 | 产品集成证据 | 源码证据：大量 channels/gateway | 源码证据：platform registry/gateway |
| provider/backend 切换 | 未见公开证据 | 主要在 Anthropic 产品边界内 | 主要在 OpenAI/Codex 产品边界内 | README/配置层有模型 provider 相关能力 | 源码证据：provider overlays + terminal backends |
| 自改进 memory/skill loop | 未见公开证据 | 有 memory/skills，但本文不推断自改进 loop | 有 memory/skills，但本文不推断自改进 loop | 有 skills/本地状态，但本文不推断自改进 loop | README + 源码支持 memory/skills/runtime |
| 研究轨迹/训练数据生成 | 未见公开证据 | 未作为本文重点 | 线程/rollout 存储相关能力 | 未作为主要定位 | README 声称 batch trajectory，源码有工具/runtime 基础 |

## Pi 具体差在哪里

### 1. 缺少可验证的执行层

Agent framework 的底层不是聊天，而是执行：读取文件、修改文件、运行命令、观察结果、继续计划。Codex 的 `apply_patch`、`request_permissions`、thread-store 和 MCP runtime，OpenClaw 的 `system.run`/sandbox，Hermes 的 terminal tool/backends 都是执行层。Pi 的公开资料没有展示同类能力。

### 2. 缺少权限与沙箱模型

一旦 agent 能执行命令，安全边界就是第一等能力。Codex 有 permission profile 和 request_permissions；OpenClaw 有 exec policy、allowlist、sandbox tool policy、DM guardrail；Hermes 有 dangerous command approval、ACP permission bridge、tool loop guardrail。Pi 没有公开的 allow/ask/deny、敏感文件策略、command classifier、sandbox backend 或远程入口防护模型。

### 3. 缺少代码工作流

Claude Code 和 Codex 的目标用户就是开发者：读代码、改代码、跑测试、看 diff、git workflow、PR/review。Pi 的公开定位不是 repo-operating coding agent，所以如果拿 Pi 做代码任务，它更像“问答/解释器”，不是能端到端落地改 repo 的工具。

### 4. 缺少工具协议与扩展生态

Codex 和 Hermes 都有 MCP runtime/client；Claude Code 官方文档也覆盖 MCP；OpenClaw 的工具策略和 gateway 体系支持插件/工具组。Pi 没有公开的 MCP、plugin、skills、hooks 或 organization/workspace 级工具扩展机制。

### 5. 缺少多会话、多代理和并行任务模型

Codex 有 session-scoped `AgentControl` 和 `AgentRegistry`；OpenClaw 有 `sessions_spawn` 和多 agent routing；Hermes 支持 concurrent tool dispatch、delegate/subagent 类能力。Pi 没有公开的 thread tree、subagent、parallel workstream 或 task queue。

### 6. 缺少常驻 gateway 和远程入口治理

OpenClaw/Hermes 都能作为 gateway 运行，接 Telegram/Slack/Discord/WhatsApp/Signal/Feishu 等渠道，还要处理 allowlist、pairing、rate limit、delivery context。Pi 有 app/web/voice，但没有公开的用户自控 gateway、session routing、channel adapter registry 或 DM 安全策略。

### 7. 缺少 automation / cron

OpenClaw 和 Hermes 都有 cron 代码；Claude Code/Codex 也有官方层面的 background/scheduled/automation 能力。Pi 没有公开的 schedule、event trigger、resumable job 或 unattended execution runtime。

### 8. 缺少 operational memory 和 skill state

Pi 可能有聊天历史或个性化，但本文没有看到它公开支持“项目记忆、工具调用日志、session search、skill 文件、自我沉淀流程、跨会话工作流复用”。Hermes 在这点最强，Codex/Claude Code/OpenClaw 也都有不同形式的 instructions/skills/memory。

### 9. 缺少 provider/backend 可控性

Hermes 的 provider registry 和 terminal backend 是代码级可见的；OpenClaw 也有 gateway/config/plugin 体系。Pi 没有公开的模型 provider 选择、自托管 endpoint、terminal backend 或 runtime backend 切换。

## 分对象对比

### Pi vs Claude Code

Claude Code 的确定优势是 coding product surface：终端、IDE、GitHub、文件编辑、命令执行、git workflow、权限、MCP、hooks、skills、subagents 等。由于 Claude Code 实现不是开源，本文不对其内部设计做源码级评价。和 Pi 比，关键差距仍然清楚：Claude Code 被公开设计为 coding agent，Pi 没有公开展示 coding-agent runtime。

### Pi vs Codex

Codex 是最适合做代码级对比的对象，因为源码里能看到 agent runtime 的骨架：thread-store、AgentControl、AgentRegistry、apply_patch、request_permissions、MCP runtime、hooks、skills、sandbox/exec 相关 crate。Pi 没有公开对应模块。换句话说，Codex 的“agent”是系统架构，Pi 的“assistant”是用户体验定位。

### Pi vs OpenClaw

OpenClaw 更像 Pi 可能想象中的“个人 AI 助手”，但它把个人助手做成了 local-first gateway：多渠道、设备节点、voice/canvas、session routing、工具执行、sandbox、cron、pairing/allowlist。Pi 也有个人助手和语音属性，但没有公开的 gateway runtime。

### Pi vs Hermes Agent

Hermes 更偏 agent OS / research runtime：模型 provider 可换，terminal backend 可换，memory/skills 可沉淀，gateway 可多渠道，cron 可无人值守，MCP 可扩展，还强调 trajectory。Pi 与 Hermes 的差距最大，因为 Hermes 试图把“长期运行、持续学习、工具执行”都放进运行时，而 Pi 的公开定位仍是对话产品。

## 不编造事实边界

- 本文不声称 Pi 没有记忆、不能联网、不能上传文件或内部没有工具；只说本次没有看到公开 agent-runtime 证据。
- 本文不声称 Claude Code 开源；本机安装包是二进制分发，不能像 Codex/OpenClaw/Hermes 一样审实现。
- 本文不把 README claim 当生产质量保证。OpenClaw/Hermes 的能力需要安装、配置、运行测试后才能评价稳定性和安全性。
- 本文不比较模型智力或回答质量，只比较产品/runtime 功能。
- 本文不评价这些工具“谁更安全”。能执行工具的 agent 风险更高，安全性要看默认策略、配置、部署和实际审计。

## 如果 Pi 想补齐到 agent framework

最小路线图大致是：

1. 工作区连接：读/search 项目文件、理解 repo 结构、维护项目 instructions。
2. 可控编辑：以 patch/diff 方式修改文件，支持 review、revert 和冲突处理。
3. 命令执行：运行测试、lint、脚本，具备 timeout、日志、工作目录和环境变量治理。
4. 权限系统：allow/ask/deny、敏感路径拦截、命令解析、审批持久化和组织策略。
5. 沙箱系统：本地受限 sandbox、云环境、网络开关、文件系统映射。
6. 工具协议：MCP 或等价 connector，接 GitHub、Slack、Jira、Drive、数据库、浏览器、observability。
7. skills/hooks/plugins：可复用工作流、生命周期 hooks、团队级扩展包。
8. 多代理：subagent、并行任务、task queue、completion notification。
9. automation：cron、事件触发、后台任务、可恢复状态。
10. gateway：多渠道消息入口、pairing/allowlist、rate limit、delivery context。
11. operational memory：项目记忆、session search、工具历史、skill 自更新。
12. developer workflow：git branch/commit/PR/review/CI/CD 集成。

## 实用建议

如果目标是聊天、解释、陪伴、语音交流，Pi 可以作为个人 AI 助手使用。

如果目标是“让 agent 真正改 repo、跑命令、提交 PR、长期自动化”，Pi 目前不应作为 framework 选型。直接 coding agent 更应该看 Claude Code 或 Codex；个人助理 gateway 更应该看 OpenClaw；自改进、provider/backend 灵活性和 research trajectory 更应该看 Hermes Agent。

## 资料链接

- Inflection AI: https://inflection.ai/
- Pi: https://pi.ai/
- Pi Help Center: https://help.pi.ai/
- Claude Code overview: https://docs.anthropic.com/en/docs/claude-code/overview
- Claude Code npm package: https://www.npmjs.com/package/@anthropic-ai/claude-code
- OpenAI Codex repo: https://github.com/openai/codex
- OpenAI Codex docs: https://developers.openai.com/codex
- OpenClaw repo: https://github.com/openclaw/openclaw
- Hermes Agent repo: https://github.com/NousResearch/hermes-agent
