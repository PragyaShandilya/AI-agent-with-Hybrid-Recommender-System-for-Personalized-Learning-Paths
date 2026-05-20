# AI-agent-with-Hybrid-Recommender-System-for-Personalized-Learning-Paths
AI-powered course recommendation system integrated with an N8N-based AI Agent that generates personalized learning paths using hybrid recommendation techniques (content-based + collaborative filtering). Built with Python, FastAPI, TF-IDF, cosine similarity, and machine learning workflows for real-time course recommendations.

## Overview

This project presents an AI-powered personalized learning recommendation system that combines a Hybrid Recommender Engine with an AI Agent workflow to generate customized learning paths for users based on their skills, interests, and proficiency levels.

The system integrates:

* **Content-Based Filtering**
* **Collaborative Filtering**
* **Natural Language Query Understanding**
* **AI-Agent Orchestration**
* **LLM-based Context Validation**

Unlike traditional recommendation systems that rely only on keyword similarity or user interactions, this project introduces an intelligent orchestration layer using AI agents to interpret learner intent, retrieve relevant courses, and validate recommendations contextually.

---

## Problem Statement

Online learning platforms provide thousands of courses across multiple domains. However, learners often struggle to:

* Find courses matching their current skill level
* Identify relevant learning paths
* Filter high-quality content from massive repositories
* Receive personalized recommendations through natural language interaction

Traditional recommender systems suffer from:

* Cold-start problems
* Lack of contextual understanding
* Limited personalization
* Weak handling of natural language queries

This project addresses these challenges by combining hybrid recommendation techniques with AI-agent reasoning.

---

# System Architecture
<img width="544" height="612" alt="image" src="https://github.com/user-attachments/assets/e134af66-08a4-4fd8-877e-085b6081e6c0" />

The system consists of three major layers:

## 1. AI Agent Layer
<img width="818" height="419" alt="image" src="https://github.com/user-attachments/assets/a80e00ff-8e5b-42d5-8f0a-a4f457cf4e24" />

The AI agent processes natural language user queries and extracts:

* Skills/domain
* Intended learning topic
* Difficulty level

Example:

```json
{
  "skills": "Machine Learning",
  "difficulty_level": "Intermediate"
}
```

This layer uses:

* N8N workflow orchestration
* LLM-based intent extraction
* Query routing
* Response validation

---

## 2. Hybrid Recommendation Engine

The recommendation engine combines:

* **Content-Based Filtering**
* **Collaborative Filtering**

### Content-Based Filtering

Uses:

* TF-IDF Vectorization
* Cosine Similarity

to compute semantic similarity between:

* User skills/interests
* Course metadata

### Collaborative Filtering

Incorporates:

* User interaction patterns
* Behavioral relevance
* Course popularity and ratings

The final recommendation score is generated using weighted score fusion.

---

## 3. Context Validation Layer

A second LLM validates recommendations by:

* Checking contextual relevance
* Matching learner proficiency
* Ensuring pedagogical consistency
* Filtering irrelevant recommendations

This creates a more reliable recommendation pipeline.

---

# Features

* Personalized learning path generation
* Natural language query handling
* AI-agent orchestration
* Hybrid recommendation system
* Difficulty-level inference
* Real-time recommendation ranking
* Context-aware validation
* API-based architecture
* Workflow automation with N8N
* Scalable modular pipeline

---

# Tech Stack

## Languages & Frameworks

* Python
* FastAPI
* N8N

## Libraries

* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

## AI / ML

* Machine Learning
* Hybrid Recommendation Systems
* LLM-based Intent Extraction
* Recommendation Ranking

---

# Workflow

## Step 1 — User Query

User enters a natural language query such as:

> "Suggest intermediate machine learning courses"

---

## Step 2 — Intent Extraction

The AI Agent extracts:

* Skill domain
* Difficulty level

using an LLM prompt-engineering pipeline.

---

## Step 3 — Recommendation Retrieval

The FastAPI backend sends structured parameters to the recommendation engine.

The engine:

* Computes semantic similarity
* Applies collaborative ranking
* Generates top-N recommendations

---

## Step 4 — Context Validation

A second AI agent validates:

* Recommendation relevance
* Skill alignment
* Difficulty suitability

---

## Step 5 — Final Response

Validated recommendations are returned to the user.

---

# Recommendation Methodology

## TF-IDF Vectorization

Course metadata including:

* Course title
* Skills
* Difficulty level

are transformed into numerical feature vectors.

---

## Cosine Similarity

Similarity is computed between:

* User query vectors
* Course vectors

to retrieve semantically relevant courses.

---

## Hybrid Scoring

Final score:

```python
score = (
    0.7 * similarity +
    0.3 * normalized_rating
)
```

This balances:

* Semantic relevance
* Course quality/popularity

---

# API Architecture

## FastAPI Endpoint

### POST `/recommend`

### Input

```json
{
  "skills": "Python",
  "level": "Beginner"
}
```

### Output

```json
[
  {
    "Course Name": "Python for Beginners",
    "Difficulty Level": "Beginner",
    "Course Rating": 4.7
  }
]
```

---

# Datasets Used

The project combines datasets from:

* Udemy
* Coursera

Course metadata includes:

* Course titles
* Skills
* Ratings
* Difficulty levels
* Descriptions

Preprocessing includes:

* Missing value handling
* Duplicate removal
* Difficulty normalization
* Rating normalization

---

# Performance & Evaluation

## AI Agent Metrics

* Difficulty-level inference accuracy: **78.95%**
* Successfully evaluated across **133 learner queries**

## System Latency

Average end-to-end latency:

```text
47.59 seconds
```

The orchestration workflow includes:

* Intent extraction
* Backend invocation
* Recommendation validation
* Response formatting

---

# Project Highlights

* Integrated AI agents with recommendation systems
* Built a modular recommendation architecture
* Implemented real-time recommendation ranking
* Automated workflows using N8N
* Designed API-based recommendation serving
* Combined semantic similarity with collaborative signals
* Added contextual reasoning using LLMs

---

# Future Improvements

* Knowledge graph integration
* Dynamic learner modeling
* Reinforcement learning for adaptive learning paths
* Lower latency using lightweight LLMs
* Real-time user feedback integration
* Improved collaborative filtering with larger interaction datasets

---
