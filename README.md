# GeoGuesser-RAG – AI Applications Project

## Project Description

GeoGuesser-RAG is a Retrieval-Augmented Generation (RAG) application as a game. Users receive AI-generated hints about a country (retrieved from a geo-dataset) and must guess the correct country. The architecture allows adding more game modes (e.g., flag guessing, 5 questions, etc.) later.

---

### Name & URL

| Name        | URL (example)                                  |
|-------------|------------------------------------------------|
| Huggingface | https://huggingface.co/spaces/YOUR-USER/GeoGuesser-RAG |
| Code        | https://github.com/YOUR-USER/GeoGuesser-RAG    |

---

## Data Sources

| Data Source    | Description                                         |
|----------------|-----------------------------------------------------|
| Custom CSV / Markdown | Country facts for game hints (name, capital, population, fun fact, [optional: flag URL]) |
| Optionally: CIA World Factbook, Wikipedia | Official global country data – to be added for larger dataset |

---

## RAG Improvements

| Improvement        | Description                                              |
|--------------------|---------------------------------------------------------|
| Query Expansion    | User questions/hints are expanded for better retrieval  |
| Result Re-Ranking  | Answers ranked by relevance to user query               |
| Optional: More    | E.g., personalized hints, difficulty levels             |

---

## Data Chunking

| Method                   | Configuration                     |
|--------------------------|-----------------------------------|
| Character/Text splitting | 500–1000 characters, 100 overlap  |

---

## Choice of LLM

| Model                | Link/Description           |
|----------------------|---------------------------|
| OpenAI GPT-3.5/4, or Hugging Face Model | [Model Overview](https://huggingface.co/models) |

---

## Testing Method

- Manual: User plays the game and judges correctness of hints.
- Optionally: Automated tests with preset queries and answers.

---

## Results

| Method                       | Accuracy | Precision | Recall |
|------------------------------|----------|-----------|--------|
| Baseline config              |          |           |        |
| With Query Expansion         |          |           |        |
| With Result Re-Ranking       |          |           |        |

---

## References

- [CIA World Factbook](https://www.cia.gov/the-world-factbook/)
- [GeoNames.org](https://www.geonames.org/)
- [Wikipedia: List of countries](https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations))
- [Hugging Face RAG Examples](https://huggingface.co/docs/transformers/main/en/tasks/rag)
- Other sources as needed
