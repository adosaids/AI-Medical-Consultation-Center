# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Medical Consultation Center (医宝医院) - A health management system based on LLMs for intelligent medical consultation. The system uses a multi-agent architecture to collect patient information, perform diagnostic reasoning, and generate treatment plans.

**Disclaimer**: This project is for educational/research purposes only and does not constitute medical advice.

## Development Commands

### Backend (Python)

```bash
# Start the FastAPI server with auto-reload
python main.py

# Or run with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (React + TypeScript + Vite)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run linter
npm run lint
```

### Environment Setup

The application requires these environment variables:
- `ERNIE_API_KEY` - Baidu Qianfan API key
- `ERNIE_SECRET_KEY` - Baidu Qianfan Secret key

## Architecture Overview

### Multi-Agent System Flow

```
User Input
    ↓
Nurse Agent (护士) - Patient interface, collects symptoms via ErnieChineseAgent
    ↓
Diagnostic Reasoning (诊断推理) - Two-agent roleplay:
  • Reasoning Expert (推理专家) - Analyzes symptoms, proposes hypotheses
  • Specialist Doctor (专科医生) - Reviews, validates, requests supplementary info
    ↓
Treatment Planning (治疗规划) - Two-agent roleplay:
  • Treatment Planner (治疗规划师) - Creates treatment plan
  • Compliance Checker (合规检测) - Validates against regulations
    ↓
Final Report to User
```

### Key Data Flow

1. **PatientCase** (`patient_case.py`) - Central data structure shared across all agents:
   - `Vital_Signs`: Structured symptom info extracted from nurse conversations
   - `Diagnosis_Process`: Step-by-step reasoning with DiagnosisStep records
   - `Supplementary_Info`: User-provided additional information during diagnosis
   - `Diagnosis`: Final diagnosis result
   - `Treatment_Plan`: Final treatment plan

2. **Streaming Architecture** - All agent communication uses WebSocket streaming:
   - `StreamingWork` manages the complete diagnosis workflow
   - `StreamingRolePlaying` handles two-agent streaming roleplay
   - `step_stream()` methods yield tokens for real-time frontend display
   - Supports pausing for supplementary information via `<need_info>` tags

3. **Model Support** - Abstracted through `ModelFactory`:
   - Baidu Qianfan models (ERNIE series) via `ErnieModel`
   - DeepSeek models via same interface
   - Streaming controlled by `liu` parameter (流式输出)

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI entry point, WebSocket endpoint handlers |
| `agent.py` | `ErnieChineseAgent` - Base agent with memory management |
| `work.py` | `StreamingWork` - Orchestrates diagnosis and treatment workflow |
| `roleplay.py` | `StreamingRolePlaying` - Two-agent conversation management |
| `patient_case.py` | `PatientCase` - Shared data structure across agents |
| `diagnosis_step.py` | `DiagnosisStep`, `DiagnosisProcess` - Records reasoning steps |
| `RAG.py` | Vector retrieval using Qdrant + ERNIE embeddings |
| `ernie_model.py` | `ErnieModel` - Model backend implementation |
| `ernie_adapters.py` | Token counter and embedding adapters for CAMEL |
| `factor.py` | `ModelFactory` - Creates model instances |

### Prompt Configuration

Agent prompts are defined in JSON files under `prompt/`:
- `Role.json` - Role definitions and constraints for each agent type
- `phase.json` - Phase-specific task prompts
- `chatchain.json` - Conversation flow configuration

Agent roles: `huanzhejiekou` (护士), `zhenduantuili` (推理专家), `zhuankeyisheng` (专科医生), `zhiliaoguihua` (治疗规划师), `lunlihegui` (合规检测)

### Frontend Structure

- `ChatPage.tsx` - Main chat interface with diagnosis flow and supplementary information handling
- `KnowledgePage.tsx` - PDF upload and RAG management
- `websocket.ts` - WebSocket service with reconnection logic and global supplement handler
- `api.ts` - API client for HTTP and WebSocket communication

### Key Implementation Details

1. **Memory Management** - `ChatHistoryMemory` with 6-round context limit to prevent token overflow
2. **Token Counting** - `ERNIETokenCounter` estimates tokens using character count / 4
3. **Context Truncation** - All long texts are truncated with `MAX_SYMPTOM_LENGTH`, `MAX_DIAGNOSIS_LENGTH` limits
4. **Flow Control** - `<stop>` tag terminates agent conversations; `<need_info>` triggers supplementary info request
5. **File Storage** - PDFs processed into chunks and stored in Qdrant vector DB at `local_data/`
6. **Supplementary Information Handling** - Global handler for processing doctor's requests for additional information during diagnosis
7. **WebSocket Error Handling** - Improved error handling and reconnection logic

### WebSocket Message Types

**Client → Server:**
- `chat` - Send chat message
- `start_diagnosis` - Begin diagnosis workflow
- `submit_supplementary_info` - Provide additional info during diagnosis

**Server → Client:**
- `chat_chunk` / `chat_complete` - Streaming chat response
- `diagnosis_phase_start` / `diagnosis_phase_complete` - Phase boundaries
- `diagnosis_chunk` - Streaming diagnosis content
- `request_supplementary_info` - Request user input
- `supplementary_info_accepted` - Confirmation that supplementary info was received
- `diagnosis_complete` - Final results with PatientCase
