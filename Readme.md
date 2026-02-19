# 🎓 AI Engineering Knowledge Assistant

Production-grade Hybrid Retrieval-Augmented Generation (RAG) system designed to answer technical questions from local documentation using grounded, source-aware responses.

This project simulates a real-world internal AI assistant used in engineering teams, MLOps platforms, and technical knowledge bases.

---

## Overview

The AI Engineering Knowledge Assistant is a CLI-based Hybrid RAG pipeline that retrieves relevant documents from a local vector database (ChromaDB) and generates context-grounded answers using an LLM (GPT-4o-mini).

Unlike basic chatbots, this system:
- Uses real project files as a knowledge base  
- Prevents hallucination with strict context grounding  
- Tracks cost, performance, and confidence metrics  
- Implements production features such as retry logic, caching, and fallback  

---

## Key Features

### 🔍 Retrieval System
- Similarity Search (semantic retrieval)
- MMR Retrieval (diverse document selection)
- Dynamic Retriever Switching
- Source File Filtering (enterprise-style control)

### 🤖 Production AI Capabilities
- Hybrid RAG Architecture (Local Embeddings + LLM)
- Modern LCEL Runnable Chain (LangChain 2024+)
- Retry Logic with Exponential Backoff
- Safe Fallback (zero extra LLM cost)
- Query Caching System
- Confidence Scoring
- Cost Tracking per Query

### 📊 Observability & Analytics
- Response time tracking
- Token usage monitoring
- API cost tracking
- ROI analytics
- Export to JSON/CSV for analysis

---

## System Architecture

User Query (CLI)  
→ EnhancedProductionRAG (Core Engine)  
→ Dynamic Retrieval Layer (Similarity / MMR)  
→ Chroma Vector Database (Local)  
→ LCEL Runnable Chain (LangChain)  
→ OpenAI GPT-4o-mini (Generation)  
→ Answer + Sources + Metrics + Cost Tracking  

---

## Project Structure

