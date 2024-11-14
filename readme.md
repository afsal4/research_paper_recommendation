# Scholar Compass

Scholar Compass is a research paper assistance project designed to assist users in finding research papers by leveraging various methods, including context-based search, hybrid search, and TF-IDF. The system utilizes advanced embeddings and similarity measures to provide relevant paper recommendations based on user queries.

## Project Overview

The system engages users in a conversational setup, asking questions to understand the type of research paper they are looking for. Based on the user’s responses, it generates a query that guides the recommendation process. The recommendation pipeline comprises several stages, from query embedding to a dynamic RAG (Retrieval-Augmented Generation) chat that provides insights on the selected papers.

---

## Evaluation

Evaluation plays a crucial role in determining which methods yield the best results. This project’s evaluation function incorporates cocitation, relevancy, and keyword relevancy metrics, tailored to align with the core objectives of the recommendation system.

The dataset consists of abstracts from 16 influential research papers, used as queries to generate recommendations for each paper. Key metrics include:
- **Cocitation**: Frequency with which both the recommended and parent paper are cited together.
- **Context Relevancy**: Cosine similarity score derived from LLM embeddings.
- **Keyword Relevancy**: Determined by TF-IDF transformation of keywords in the query and recommended papers.

The 16x30 metric matrix is normalized and weighted for comprehensive evaluation.

---

## System Architecture

### 1. Human-LLM Interaction
The system gathers user information through a conversation, allowing the LLM to generate a query based on the user’s needs.

### 2. Query Embedding and Similarity Search
- **Embedding**: The user’s query is converted into a vector representation using Google embeddings.
- **Similarity Search**: A cosine similarity search within a vector store to identify the top 3 relevant papers.

### 3. RAG Chat
- **Recommendation Conversion**: Selected papers are vectorized for interactive QA.
- **Interactive QA Chatbot**: The RAG chatbot provides detailed insights into the recommended papers.
- **Conclusion**: Summarizes user inquiries and the provided recommendations.

---

## Evaluation Methods

### Google Embeddings
- Abstracts are chunked and embedded, achieving high cocitation and context relevancy scores.

### Hybrid Search
- Combines Google embeddings (dense) with BM25 (sparse) from Pinecone for efficient hybrid search.

### TF-IDF Search
- Sparse vector representation using a max feature setting of 15,000, preprocessed with stop word removal and lemmatization.

---

## Observations and Improvements

1. **Google Embeddings**: Good results with stop words; better relevancy in dense vector search.
2. **Hybrid Search**: Effective without chunking; benefits from combined dense and sparse representations.
3. **TF-IDF**: High keyword relevancy; effective in sparse vector search.
4. **General**: Enhanced with more training data; potential improvements with hyperparameter tuning.

---

### Live Link : [https://researchpaperrecommendation-eufsvxujfqkvecp6ovhktv.streamlit.app/](https://researchpaperrecommendation-eufsvxujfqkvecp6ovhktv.streamlit.app/)

## Conclusion

This recommendation system demonstrates the potential for efficient academic search solutions. Future enhancements could include increasing dataset diversity, optimizing TF-IDF parameters, and refining hybrid search techniques.

---

