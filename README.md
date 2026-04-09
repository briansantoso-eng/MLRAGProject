# CloudDocs RAG System

CloudDocs RAG is a practical retrieval-augmented generation project built around real cloud provider documentation. It demonstrates how to combine document ingestion, vector search, and large language models to deliver grounded answers from AWS, Azure, and GCP technical content.

## What this project does

- Ingests real AWS, Azure, and GCP documentation
- Converts documentation into searchable vector embeddings
- Stores content in a local ChromaDB vector store
- Retrieves relevant passages for user queries
- Generates grounded answers using a modern LLM
- Supports conversational follow-up with memory and query rewriting

## Why it matters

Modern enterprise search and knowledge systems need answers that are both relevant and verifiable. This project shows how to:

- reduce hallucination by grounding responses in actual documentation
- enable cloud operations teams to query provider docs quickly
- speed up troubleshooting, onboarding, and architectural research
- build a reusable RAG pipeline for internal knowledge bases

## Real-world contribution

CloudDocs RAG is useful for teams that need:

- developer support for cloud architecture decisions
- fast access to multi-cloud documentation without manual search
- a proof-of-concept for integrating LLMs with real knowledge sources
- a low-cost pipeline for searchable technical content

## Key components

- `step1_ingest.py`: fetches and processes cloud docs into structured chunks
- `step2_embed_store.py`: creates and stores vector embeddings in ChromaDB
- `step3_rag_query.py`: runs retrieval-augmented queries with grounding
- `step4_chat.py`: provides a conversational interface with context handling
- `config.py`: central configuration for models, chunking, and retrieval

## Architecture

The system connects these layers:

- source documentation -> content ingestion
- text cleaning -> chunking with overlap
- vector embedding generation -> vector database storage
- similarity retrieval -> prompt construction
- LLM generation -> grounded response output

## Practical use cases

- internal knowledge search for platform engineering teams
- QA support for cloud operations and security teams
- training datasets for enterprise RAG systems
- evaluating cloud provider feature comparisons and best practices

## Quantifiable Proof

The system successfully ingests and provides grounded answers from multi-cloud documentation. Below is a sample chat session demonstrating retrieval from AWS, Azure, and GCP sources:

```
📊 Knowledge base ready: 50 document chunks
🤖 RAG Chat initialized!
💡 Ask q what are GCP IAM roles for service accounts
📚 Sources: GCP IAM (GCP), GCP IAM (GCP), GCP IAM (GCP)
🤖 GCP IAM roles for service accounts are a crucial aspect of managing access and permissions in Google Cloud. To answer your question, GCP IAM roles for service accounts are used to define the permissions and access that a service account has to Google Cloud resources. Service accounts are used to authenticate and authorize access to Google Cloud services on behalf of a user or another service. In GCP, you can create custom roles for service accounts using the IAM service. This allows you to define specific permissions and access for a service account, rather than using a pre-defined role. According to the documentation, you can create custom roles for service accounts by using the IAM service. You can also use the IAM client libraries to create and manage custom roles for service accounts. Here are some key points to consider when creating custom roles for service accounts in GCP: 1. **Service account roles**: You can create custom roles for service accounts using the IAM service. This allows you to define specific permissions and access for a service account. 2. **Custom roles**: You can create custom roles for service accounts using the IAM client libraries. This allows you to define specific permissions and access for a service account. 3. **Permissions**: You can define specific permissions for a service account using the IAM service or the IAM client libraries. 4. **Access**: You can define the access that a service account has to Google Cloud resources using the IAM service or the IAM client libraries. To get started with creating custom roles for service accounts in GCP, I recommend checking out the documentation on creating and managing custom roles. You can also use the IAM client libraries to create and manage custom roles for service accounts. Do you have any specific questions about creating custom roles for service accounts in GCP? 

👤 You: How does Cloud Storage lifecycle management work?
📚 Sources: GCP Storage (GCP), Azure Storage (AZURE), Azure Virtual Machines (AZURE)
🤖 You're interested in learning more about Cloud Storage lifecycle management. That's a great topic, especially considering the vast amounts of data being stored in the cloud. From what I understand, Cloud Storage lifecycle management is a feature that allows you to manage the storage and retrieval of your data in a more efficient and cost-effective way. It's essentially a set of rules that dictate how your data is stored, accessed, and deleted based on its age, usage, and other factors. In GCP, Cloud Storage lifecycle management is a feature of the Cloud Storage service. According to the documentation, you can use lifecycle management to: * Archive or delete objects after a certain period of time * Transition objects to a lower-cost storage class * Set retention policies for objects * Configure automatic deletion of objects To get started with Cloud Storage lifecycle management in GCP, I recommend checking out the documentation on managing data in Cloud Storage. You can also use the Cloud Storage client libraries to create and manage lifecycle policies for your data. Now, I'm curious - are you looking to implement lifecycle management for a specific use case, such as archiving old data or reducing storage costs? Or are you just looking to learn more about the feature in general? 

👤 You: In short answer what are the differences between Lambda and EC2 instances
📚 Sources: AWS Lambda (AWS), AWS Lambda (AWS), AWS Lambda (AWS)
🤖 You're looking for a brief comparison between AWS Lambda and EC2 instances. Both are compute services offered by AWS, but they serve different purposes and have distinct characteristics. **Lambda** is a serverless, event-driven compute service that runs code without the need to manage servers. You write and organize your code in Lambda functions, which are triggered by events or API calls. Lambda manages the underlying infrastructure, scaling up and down automatically, and charges you only for the compute time used. This makes it ideal for tasks that require short bursts of compute power, such as file processing, long-running workflows, or database operations. **EC2 instances**, on the other hand, are virtual servers that you can use to run your own applications and services. You have full control over the instance, including the operating system, software, and configuration. EC2 instances are suitable for applications that require a high degree of customization, control, or long-term compute resources. Key differences between Lambda and EC2 instances: * **Serverless vs. Server-based**: Lambda is serverless, while EC2 instances are server-based. * **Compute model**: Lambda uses a pay-per-use pricing model, while EC2 instances are billed by the hour or by the instance type. * **Scalability**: Lambda scales automatically, while EC2 instances require manual scaling or use of Auto Scaling. * **Management**: Lambda manages the underlying infrastructure, while EC2 instances require manual management and maintenance. In summary, if you need a flexible, scalable, and cost-effective solution for short-lived tasks or event-driven computing, Lambda is a great choice. If you require a high degree of customization, control, or long-term compute resources, EC2 instances might be a better fit. Do you have any specific use cases or requirements that might help me provide a more tailored comparison? 
```

![RAG Chat Screenshot](ss/ss1.png)

> Note: The screenshot file is stored at `ss/ss1.png` in this repository.

## Live Demo

Try the app live on Streamlit Cloud:

**[https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   ```
3. Provide your Groq/Open AI API

## How to run

Use the available scripts to build and query the system:

- `python step1_ingest.py`
- `python step2_embed_store.py`
- `python step3_rag_query.py`
- `python step4_chat.py`

## Benefits of this approach

- grounded answers from source documentation
- a clear vector search pipeline for cloud docs
- minimal cost with local embeddings and efficient LLM use
- extensible design for additional providers and content types

---
