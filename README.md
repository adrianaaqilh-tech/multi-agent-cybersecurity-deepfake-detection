# Multi-Agent System for Cybersecurity Threat and Deepfake Detection

## Project Overview

This project is a Multi-Agent System for Cybersecurity Threat and Deepfake Detection. The system uses multiple specialised agents to analyse cybersecurity threats and detect possible deepfake images based on the user's request.

## Requirements

- Python 3.10 or above
- LM Studio
- Gemma model
- Tesseract OCR
- Git

## Installation

### 1. Clone the Repository

    git clone  https://github.com/adrianaaqilh-tech/multi-agent-cybersecurity-deepfake-detection.git
    cd multi-agent-cybersecurity-deepfake

### 2. Create a Virtual Environment

    python -m venv .venv

Activate the virtual environment:

**Windows PowerShell:**

    .venv\Scripts\Activate.ps1

### 3. Install Required Packages

    pip install -r requirements.txt

### 4. Install Tesseract OCR

Install Tesseract OCR on Windows.

The system expects Tesseract to be installed at:

    C:\Program Files\Tesseract-OCR\tesseract.exe

Verify the installation using:

    & "C:\Program Files\Tesseract-OCR\tesseract.exe" --version

### 5. Install and Configure LM Studio

Install LM Studio and download the required Gemma model.

The project currently uses:

    google/gemma-4-e4b

Start the LM Studio server using:

    lms server start

Check the server status using:

    lms server status

The system uses the following local API endpoint:

    http://localhost:1234/v1

## Running the System

After completing the installation and configuration, run the application using:

    python app.py

The system will request a user prompt and an image input before starting the multi-agent workflow.

## Example Requests

### Cybersecurity Analysis

    Check this image for cybersecurity threats.

The Dynamic Router routes the request to the **Cybersecurity Agent**.

### Deepfake Analysis

    Is this image a deepfake?

The Dynamic Router routes the request to the **Deepfake Agent**.

### Combined Analysis

    Check this image for cybersecurity threats and deepfake manipulation.

The Dynamic Router routes the request to both the **Cybersecurity Agent** and **Deepfake Agent**.

### Out-of-Scope Request

    Can you give me a recipe for this cooking?

The request is classified as **out-of-scope** and no specialist analysis is performed.

## Multi-Agent Analysis

The system consists of several components:

- **Dynamic Router** – determines the type of analysis required from the user's request.
- **Cybersecurity Agent** – analyses possible cybersecurity threats such as phishing and suspicious URLs.
- **Deepfake Agent** – analyses image indicators related to possible AI-generated or manipulated images.
- **Evidence Agent** – reviews the specialist results and produces a consolidated assessment.
- **OCR Tool** – extracts text from uploaded images and provides supporting context for the analysis.

## Conditional Routing

The Dynamic Router supports different routing paths:

                     User Request
                          ↓
                   Dynamic Router
                          ↓
                 Conditional Routing
                          ↓
     ┌───────────────┬───────────────┬───────────────┐
     ↓               ↓               ↓               ↓
    Cybersecurity   Deepfake        Both          Out of    
    Agent           Agent           Agents         Scope
     └───────────────┴───────────────┴───────────────┘
                          ↓
                    Evidence Agent
                          ↓
                   Final Assessment

## Robustness and Fallback

The system includes several mechanisms to improve reliability:

- LLM timeout and retry configuration
- Exception handling
- JSON output validation
- Workflow iteration limit
- LLM availability checking
- Fallback routing when the local LLM is unavailable

If the LLM service is unavailable, the system can use a fallback routing mechanism instead of stopping the entire workflow.

## Project Structure

    multi-agent-cybersecurity-deepfake/
    │
    ├── agents/
    │   ├── router.py
    │   ├── cybersecurity_agent.py
    │   ├── deepfake_agent.py
    │   └── evidence_agent.py
    │
    ├── tools/
    │   ├── ocr_tools.py
    │   ├── cybersecurity_tools.py
    │   ├── deepfake_tools.py
    │   └── evidence_tools.py
    │
    ├── workflow/
    │   └── graph.py
    │
    ├── app.py
    ├── config.py
    ├── state.py
    ├── requirements.txt
    └── README.md

## GitHub Repository

The complete source code is available in the project's GitHub repository:

    https://github.com/adrianaaqilh-tech/multi-agent-cybersecurity-deepfake-detection.git

## Notes

- Make sure LM Studio is running before executing the MAS.
- Make sure the required Gemma model is available in LM Studio.
- Make sure Tesseract OCR is installed at the expected location.
- Do not upload API keys, passwords, or other sensitive information to GitHub.
- The `.venv` folder should not be included in the repository.

## Stop the LM Studio Server

After finishing the testing, the LM Studio server can be stopped using:

    lms server stop