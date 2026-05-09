---
name: lcp-pinn-project-guardrails
description: Use for all work inside this LCP-PINN project. The assistant must not modify project files by default, and should provide guidance, explanations, checklists, and copyable pseudocode unless the user explicitly asks the assistant to write or edit code.
---

# LCP-PINN Project Guardrails

## Core Rule

In this project, do not modify, create, delete, or patch any project files by default.

Only edit files when the user explicitly asks you to write, modify, generate, delete, or patch code/files on their behalf.

## Default Behavior

When helping with PINN, LCP, physical modeling, training logic, debugging, or project structure:

- Explain concepts clearly in Chinese by default.
- Point to the relevant files, functions, variables, and shapes to inspect.
- Provide step-by-step reasoning and debugging checks.
- Provide pseudocode or code-like blocks that the user can copy manually.
- Prefer small, inspectable steps over large implementations.
- Ask the user to confirm before changing files if their request is ambiguous.

## Allowed Without Extra Confirmation

The assistant may:

- Read files to understand the current project.
- Run non-destructive inspection commands.
- Explain formulas, tensor shapes, network architecture, and loss design.
- Give copyable pseudocode blocks.
- Review user-written code and suggest changes.

## Not Allowed Unless Explicitly Requested

Do not:

- Edit Python files.
- Edit YAML, Docker, config, or dataset files.
- Add new implementation files.
- Delete project files.
- Apply patches.
- Refactor code.
- Write complete runnable implementations.

If the user asks for help implementing something but does not clearly ask the assistant to edit files, respond with guidance and pseudocode instead of modifying the repository.

## Explicit Permission Examples

Editing is allowed when the user says things like:

- "你来改这个文件"
- "直接修改"
- "帮我写到文件里"
- "替我实现"
- "可以改代码"
- "生成这个 Python 文件"
- "删除这个文件"

Even with explicit permission, keep changes minimal, localized, and easy for the user to inspect.

