# 🧠 DocAgent

**DocAgent** is a lightweight, modular document assistant powered by LangChain and OpenRouter. It allows you to semantically search and summarize educational or technical documents using cutting-edge language models like Mistral-7B.

Built to experiment with vector search, embeddings, and LLM orchestration via LangChain — this project is API-key agnostic and supports pluggable models through OpenRouter.

---

## 🚀 Features

- 🔎 **Semantic Search** with score filtering and top-k ranking
- 📄 **Smart Summarization** with adjustable summary length (short, medium, long)
- ⚙️ **Vector Store Stats** to monitor FAISS-based document index
- 🧠 **Singleton-based Model Manager** for efficient caching and reuse
- 🔌 **Flexible API Support** — compatible with any OpenRouter key or HuggingFace embeddings

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) — LLM tooling & agent support  
- [FAISS](https://github.com/facebookresearch/faiss) — fast document similarity search  
- [OpenRouter](https://openrouter.ai/) — unified access to hosted language models  
- [HuggingFace Transformers](https://huggingface.co/) — embedding provider  
- Python 3.10+

---




## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

You’ll need an [OpenRouter](https://openrouter.ai/) API key to use LLM features. You can pass it manually when calling the summarization tool, or inject via environment variable:

```bash
export OPENROUTER_API_KEY=your-api-key
```

---


## 🙏 Thanks for Visiting

Thank you for taking the time to explore this project.  
If you have any questions, feedback, or ideas — feel free to reach out or open an issue.


📬 Let’s connect and build something great!

[![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/nazaninghobadi)
[![Telegram](https://img.shields.io/badge/Telegram-26A5E4?logo=telegram&logoColor=white)](https://t.me/your_telegram_username)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:your.email@example.com)

