# **SpecSentinel: Detecting Inconsistencies in Programming Language Specifications Using AI**

---

## 📌 Problem Statement

Programming language specifications (like the Java Language Specification and JVM Specification) are complex, semi-formal documents that evolve over time. They are often:

- Written in natural or semi-structured language  
- Maintained across versions with evolving behavior  
- Prone to **inconsistencies**, **ambiguous semantics**, or **conflicting rules** (e.g., in method resolution, generics, exception handling)

These inconsistencies can lead to:

- Misunderstandings by compiler/toolchain developers  
- Variability in implementation across JVMs  
- Developer confusion and subtle bugs in edge cases

---

## 🚀 Innovation: Why This Is a Novel Idea

While formal verification and compiler testing exist, **there is no tool that analyzes natural-language specifications themselves** for contradictions or drift across versions.

**SpecSentinel** introduces a new paradigm:

- **LLM-powered semantic understanding** of English specification prose  
- **Symbolic reasoning** to detect logical contradictions or ambiguities  
- **Spec version drift detection** (e.g., JLS 8 → 11 → 17 → 21)  
- **Specification linting** — analogous to code linters, but for the specs themselves

This idea is inspired by research like *CellularLint* but targets a different and previously unexplored domain.

---

## ⚙️ How It Works

### 1. Spec Ingestion

- Input: Java Language Specification (JLS) and JVM Specification in HTML or PDF form  
- NLP-based preprocessing to:
  - Segment the spec into discrete rules and behaviors  
  - Normalize structure for rule modeling

### 2. Behavior Modeling Engine

- Extract rules for:
  - Method resolution, overloading  
  - Type coercion and compatibility  
  - Inheritance and overriding behavior  
- Use an LLM (via OpenRouter API) to convert rules into symbolic logic

> **Note:** You must provide a valid `OPENROUTER_API_KEY` to use LLM services.

### 3. Inconsistency Detection

- Translate rules into logical expressions  
- Use the **Z3 SMT Solver** to check for:
  - Contradictions  
  - Ambiguities  
  - Redundancies or unreachable rules  
- Output conflict reports with **graph-based visualizations**

### 4. Version Drift Analyzer

- Compare semantically equivalent rules across JLS versions  
- Highlight:
  - Added/removed/modified rules  
  - Potential breakage or undefined behavior  
  - Semantic drift over time

---

## 🧪 Running the Project

This project is intended to be run in **Google Colab**.

### ✅ Prerequisites

- A Google account (for Colab and Drive access)
- A valid `OPENROUTER_API_KEY` for LLM access
- A copy of the Java Language Specification (HTML or PDF)

### 📁 File Storage

All generated artifacts (parsed rules, visualizations, intermediate outputs) are saved to **Google Drive** for persistence across sessions.

### ▶️ Steps to Run

1. Open the Colab notebook (`specsentinel.ipynb`)
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
3. Set your OpenRouter API key:
    ```python
    import os
    os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"
4. Run through each cell to:
    - Preprocess spec
    - Extract rules
    - Analyze and detect contradictions
    - Generate and store outputs
