# GeoGuesser-RAG – AI Applications Project
This documentation describes the architecture, methodology, evaluation, and learnings from the GeoGuesser-RAG AI quiz project, developed as part of the AI Applications module.

## Project Description

GeoGuesser-RAG is an interactive, Retrieval-Augmented Generation (RAG) application designed as an educational and entertaining quiz game. All factual data, such as country facts, capitals, flags, and food, is sourced from a custom CSV dataset, which serves as the knowledge base for the RAG pipeline.

At the heart of the system is Google Gemini, a large language model (LLM), integrated via LangChain. The LLM generates user-facing hints and answers by combining retrieved, context-rich chunks from the knowledge base with generative capabilities.

The application currently supports three engaging game modes:
1. **Hint Game:** Users receive dynamically generated hints about a country and must guess which country is being described.
2. **Flag Quiz:** Players are shown a flag image and try to identify the corresponding country.
3. **Capital Quiz:** The system presents a capital city, and users must provide the correct matching country.

Key features and innovations include:
- **End-to-end RAG pipeline:** Combining CSV-based structured data with advanced semantic retrieval and LLM-powered generation.
- **Modular, extensible architecture:** Easily expandable to include additional game modes or new data types in the future.
- **Retrieval improvements:** Implements query expansion and category-based re-ranking for more accurate, context-sensitive hint generation.
- **Visual analytics:** Includes UMAP-based embedding visualizations for debugging and evaluation of the retrieval process.

GeoGuesser-RAG showcases how modern RAG architectures can be applied to educational games, offering a flexible foundation for both learning and entertainment with room for future enhancements.


---

### Name & URL

| Name        | URL (example)                                  |
|-------------|------------------------------------------------|
| Huggingface | https://huggingface.co/spaces/selintyrs/GeoGuesser |
| Code        | https://github.com/selintyrs/geoguesser   |

---

## Data Sources

### Data Sources

The game relies on a consolidated country dataset compiled from several reputable open data sources. These include official statistics and curated lists for facts such as capitals, flags, national dishes, and languages. All listed sources (see table above) were individually downloaded and programmatically merged in a dedicated data preparation notebook.

During the preparation phase, the raw datasets were **cleaned, normalized, and merged**, resolving differences in naming conventions and filling in missing entries where possible. Additional attributes were added to enrich the dataset, ensuring that each country entry contains all relevant facts in a unified structure.

The result of this data engineering process is a single, clean CSV file (`countries_final.csv`) that aggregates **all necessary information for each country**—including name, capital, flag URL, main dishes, official languages, and more. This unified dataset serves as the foundation for all retrieval and generative processes within the GeoGuesser-RAG application.


| Data Source | Description |
|-------------|-------------|
| [CIA World Factbook](https://www.kaggle.com/datasets/lucafrance/the-world-factbook-by-cia) | Official global country data – to be added for larger dataset |
| [Most Popular National Dishes by Country](https://worldostats.com/country-stats/popular-foods-by-country/) | National dishes for each country |
| [Countries – WORLDOSTATS (Flags)](https://worldostats.com/countries/) | Flag images for countries |
| [Official Languages by Country (2025) \| Global Language List](https://worldostats.com/country-stats/official-language-by-country/) | Official languages by country |
| [Country Capitals in 2025 \| Full List by Country](https://worldostats.com/country-stats/capitals-by-country/l) | Official country capitals 2025 |

---

## RAG Improvements


To maximize the effectiveness of retrieval-augmented generation in GeoGuesser-RAG, several targeted improvements were implemented:


| Improvement                      | Description                                                                                  |
|----------------------------------|---------------------------------------------------------------------------------------------|
| Structured Document Generation   | All facts for each country are combined into a structured, human-readable text document, making retrieval more consistent and boosting LLM performance on answer synthesis.           |
| Query Expansion                  | User questions/hints are expanded into natural language prompts, improving retrieval recall. |
| Category-Based Result Re-Ranking | Retrieved answers are re-ranked based on category-specific relevance to the user’s query.    |
| Embedding Visualization (UMAP)   | Embeddings are projected into 2D using UMAP for cluster analysis and debugging, helping to verify that similar facts are grouped together and the retrieval space is healthy.|



---

## Data Chunking

To enable efficient retrieval and ensure high-quality responses from the language model, the country dataset was divided into manageable text segments (“chunks”). Two chunking strategies were used: character-based splitting with overlap to preserve context, and token-based splitting aligned with the model’s tokenizer. This approach helps maintain semantic coherence and maximizes retrieval performance for both short and long queries.


| Method                   | Configuration                                        |
|--------------------------|------------------------------------------------------|
| Character/Text splitting | 1000 characters per chunk, 100 character overlap     |
| Token-based splitting    | 128 tokens per chunk, using model-specific tokenizer |


---

## Choice of LLM
For this project, Google Gemini 1.5 Pro was selected as the main language model. Gemini is particularly well-suited for Retrieval-Augmented Generation (RAG) applications due to its advanced reasoning capabilities, large context window, and support for complex multi-turn instructions. Its multimodal capabilities enable it to understand nuanced prompts and synthesize context-rich hints from retrieved country facts. This makes Gemini a strong choice for delivering high-quality, diverse, and accurate hints in a GeoGuesser-like educational game, where the model must adapt to various topics (history, food, geography) and generate engaging, natural language clues based on retrieved context.


| Model                           | Link/Description                                                                                         |
|----------------------------------|---------------------------------------------------------------------------------------------------------|
| Google Gemini 1.5 Pro           | [Gemini API – Model Overview](https://ai.google.dev/gemini-api/docs/models?hl=en#gemini-1.5-pro) <br>Multimodal, advanced reasoning, large context window. Used via LangChain integration in this project. |

---

## Testing Method

The system was evaluated using both manual and automated methods:

- **Manual Testing:** The GeoGuesser game was played interactively to subjectively assess the quality of generated hints, flag, and capital evaluations.
- **Automated Testing:** Dedicated unit tests were implemented for each module (Hint, Flag, Capital). For Hints, automated pipelines compared different retrieval configurations: Baseline (no query expansion, no re-ranking), Query Expansion, and Re-Ranking. Each test used a set of countries, categories, and expected keywords to objectively measure accuracy, precision, and recall.

For Hints, three retrieval pipelines were compared:
1. **Baseline:** Standard retrieval without any enhancements.
2. **Query Expansion:** The user query is expanded to include synonyms and related terms before retrieval.
3. **Re-Ranking:** Retrieved results are re-ranked based on additional relevance scoring.

Flags and Capitals were evaluated on exact matches and did not require alternative pipelines.

---

## Results

The following table summarises the evaluation metrics for all tested components and retrieval pipelines in the GeoGuesser-RAG project. The results are based on automated tests using a set of diverse countries and categories. For Hint generation, we compared three retrieval strategies: baseline retrieval, query expansion, and category-based re-ranking. For Flag and Capital questions, exact matching was used.

Accuracy, precision, and recall were calculated using `sklearn.metrics` to objectively assess system performance. These metrics provide a transparent overview of the strengths and current limitations of each module, and form the basis for future improvements.


| Method                        | Accuracy | Precision | Recall |
|-------------------------------|----------|-----------|--------|
| Baseline (Hint)               |   0.27   |   0.27    |  0.27  |
| Query Expansion (Hint)        |   0.47   |   0.47    |  0.47  |
| Re-Ranking (Hint)             |   0.27   |   0.27    |  0.27  |
| Flag (Exact Match)            |   1.00   |   1.00    |  1.00  |
| Capital (Exact Match)         |   0.53   |   1.00    |  0.53  |

---

**Interpretation:**

- **Query Expansion** improved Hint retrieval, demonstrating that enriching user queries with synonyms and related terms leads to better information coverage and recall. However, while the metric increase is clear, some irrelevant or too generic expansions may still dilute precision, suggesting further fine-tuning potential.
- **Re-Ranking** did not outperform the baseline in this configuration. This indicates that the current re-ranking logic may not sufficiently prioritise the most relevant facts for the user’s category. Exploring more advanced or category-aware ranking strategies could help here.
- **Flag** accuracy is perfect because exact URL matches leave little room for ambiguity. This highlights that the dataset for flags is clean and well-structured, but it also means that any errors would likely stem from data issues, not the retrieval process itself.
- **Capital** tests achieved moderate accuracy and high precision, indicating the model reliably recognises correct answers but struggles with close variations or alternative spellings. Expanding accepted answers (e.g., by including common typos or alternate names) could improve usability and fairness.
- **General Observation:** The overall performance reveals that retrieval-augmented generation works reliably for factual, structured tasks (flags, capitals), while more open-ended hints benefit significantly from smarter retrieval strategies like query expansion. The strict evaluation metrics also point to the value of continuously refining both data quality and retrieval pipelines.


---
## Conclusion
The GeoGuesser-RAG project demonstrates the potential and challenges of applying retrieval-augmented generation techniques in an interactive quiz game setting. Automated evaluations show that query expansion significantly improves hint accuracy, while re-ranking in its current form does not add further benefit. Flag and capital quizzes achieve high precision due to the structured and deterministic nature of the evaluation criteria.

However, the hint generation game remains challenging for users, as the hints are not yet dynamically tailored to the player's needs or difficulty level. In future iterations, introducing adaptive, multi-step hints (e.g., from more general to more specific) could make the game more accessible and engaging for a broader audience. Additionally, improving the user interface and overall usability—such as providing clearer feedback, integrating hint selection, and supporting typo-tolerance for user inputs—would greatly enhance the player experience.

Further improvements could include:
- Expanding the test set and categories for more robust evaluation.
- Incorporating human-in-the-loop feedback to fine-tune both retrieval and hint generation.
- Exploring hybrid LLM and rules-based pipelines for edge cases where LLMs underperform.
- Adding more game modes and analytics to track user learning progress.

With these results, GeoGuesser-RAG is well positioned as a testbed for future educational and entertainment use cases leveraging Retrieval-Augmented Generation.



---
## References

- **LangChain**  
  [LangChain Documentation](https://python.langchain.com/) – A framework for developing applications powered by language models.

- **Google Gemini API**  
  [Gemini API Documentation](https://ai.google.dev/gemini-api/docs) – Multimodal large language model API for advanced AI use cases.

- **Sentence Transformers**  
  [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.sbert.net/) – Efficient sentence embeddings for semantic search and retrieval.

- **UMAP**  
  [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/) – Nonlinear dimension reduction technique for visualization and clustering.
