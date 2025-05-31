import os
import pandas as pd
import numpy as np
import random
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import gradio as gr

# --- Lade API-Key (über Hugging Face Secret) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY not set!"

# --- Daten laden ---
countries_path = "./countries_final.csv"
countries_df = pd.read_csv(countries_path)

# Preprocessing
str_cols = ['country', 'background', 'location', 'map_reference', 'area_comparative',
            'capital', 'country_clean', 'official_language', 'food1', 'food2', 'food3', 'flag_url']
for col in str_cols:
    countries_df[col] = countries_df[col].astype(str).str.strip()
for food_col in ['food1', 'food2', 'food3']:
    countries_df[food_col] = countries_df[food_col].fillna('Not specified')
countries_df['flag_url'] = countries_df['flag_url'].fillna('')
countries_df['Population'] = countries_df['Population'].apply(lambda x: int(float(x)) if x != 'nan' else 0)

# --- Lade Embeddings und FAISS-Index ---
index = faiss.read_index("country_faiss_index/index.faiss")
with open("country_faiss_index/index.pkl", "rb") as f:
    token_split_texts = pickle.load(f)

# --- Models für Embedding und LLM ---
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# --- Prompt Template für das LLM ---
system_template = """
You are an AI quizmaster. Based on the following context about the country, generate a helpful hint for the quiz player.

Category: {category}
Details: {details}

Context:
{context}

Hint:
"""
question_answering_prompt = PromptTemplate(
    template=system_template,
    input_variables=["country", "category", "details", "context"]
)

def expand_query(user_input):
    user_input = user_input.strip().lower()
    if user_input in ["geography", "geo"]:
        return (
            "Create a new, interesting geography hint as a single sentence. "
            "Combine at least two of these: neighboring countries, capital city, population, area (with comparison), "
            "region, major cities, or a notable physical feature. Use different combinations for each new hint."
        )
    elif user_input in ["flag", "flag description"]:
        return (
            "Describe one special aspect of the national flag in a clear sentence. "
            "Mention main colors and one symbol. If possible, include the flag URL."
        )
    elif user_input in ["food", "cuisine"]:
        return (
            "State two key traditional dishes or food specialty in a natural sentence. "
            "Mention the food type, how it is eaten, or its importance in culture."
        )
    elif user_input in ["history"]:
        return (
            "Write two historical facts as two sentences, each time using a new detail. "
            "Choose from founding, independence, major wars/events, famous leaders, or other historical facts."
        )
    elif user_input in ["random"]:
        return (
            "Give a surprising or fun fact about the country: people, language, tradition, nature, or records."
        )
    else:
        return f"Give a short, quiz-style fact about the country related to: {user_input}"

def rerank_chunks_by_category(chunks, hint_type, top_n=5):
    keywords_map = {
        "geography": ["border", "neighbor", "capital", "region", "area", "city", "mountain", "sea", "river"],
        "food": ["dish", "cuisine", "eat", "food", "meal", "specialty", "ingredient"],
        "flag": ["flag", "symbol", "color", "banner", "emblem"],
        "history": ["war", "independence", "colony", "leader", "founding", "republic", "revolution"],
        "random": [],
    }
    keywords = keywords_map.get(hint_type.lower(), [])
    ranked_chunks = sorted(
        chunks,
        key=lambda c: sum(1 for kw in keywords if kw in c.lower()),
        reverse=True
    )
    return ranked_chunks[:top_n]

def is_guess_close(guess, country):
    from difflib import SequenceMatcher
    guess = guess.strip().lower()
    country = country.strip().lower()
    ratio = SequenceMatcher(None, guess, country).ratio()
    return ratio >= 0.85

# ------------------- Gradio Interface -------------------
def get_continents():
    return sorted(set(countries_df['map_reference'].dropna()))

highscores = {"Hints Game": 0, "Flag Quiz": 0, "Capital Quiz": 0}

def hints_start(continent):
    countries_in_continent = countries_df[countries_df['map_reference'] == continent]
    if countries_in_continent.empty:
        return '', '', '', [], [], 1, 0, 'No country found. Please select a different continent.'
    chosen_row = countries_in_continent.sample(1).iloc[0]
    country_name = chosen_row['country']
    return country_name, '', '', [], [], 1, 0, f'Game started! Country selected. Please choose a hint category.'

def hints_get_hint(country_name, hint_type, used_hints, used_categories, hint_nr):
    if not country_name or not hint_type:
        return '', used_hints, used_categories, hint_nr, 'Please select a country and a hint category first.'
    details = expand_query(hint_type)
    llm_query = f'{country_name} {hint_type}'
    query_embedding = embedding_model.encode(llm_query)
    k = 5
    D, I = index.search(np.array([query_embedding]), k)
    retrieved_chunks = [token_split_texts[idx] for idx in I[0]]
    ranked_chunks = rerank_chunks_by_category(retrieved_chunks, hint_type, top_n=3)
    filtered_chunks = [c for c in ranked_chunks if (hint_type, c) not in used_hints]
    if not filtered_chunks:
        return '', used_hints, used_categories, hint_nr, 'No new hints available for this category. Please select another one.'
    hint_chunk = filtered_chunks[0]
    used_hints.append((hint_type, hint_chunk))
    used_categories.append(hint_type)
    context = hint_chunk
    prompt = question_answering_prompt.format(
        country=country_name,
        category=hint_type,
        details=details,
        context=context
    )
    response = llm.invoke(prompt)
    return response.content, used_hints, used_categories, hint_nr+1, f'Hint {hint_nr}: Category {hint_type}'

def hints_check_guess(user_guess, country_name, highscore):
    user_guess_display = user_guess or "[empty]"
    if not user_guess or not country_name:
        return highscore, f'❌ Incorrect! Your guess: {user_guess_display} | Correct answer: {country_name}', '', ''
    if user_guess.strip().lower() == country_name.strip().lower():
        highscore += 1
        highscores["Hints Game"] = max(highscore, highscores["Hints Game"])
        return highscore, f'✅ Correct! Your guess: {user_guess_display} | Country: {country_name}', '', ''
    else:
        return highscore, f'❌ Incorrect! Your guess: {user_guess_display} | Correct answer: {country_name}', '', ''

def flag_start(continent):
    countries_in_continent = countries_df[
        (countries_df['map_reference'] == continent) &
        (countries_df['flag_url'].notnull()) &
        (countries_df['flag_url'].str.strip() != "")
    ]
    if countries_in_continent.empty:
        return '', '', '', 0, 0, 'No country with a flag found. Please select a different continent.'
    country, flag_url = random.choice(countries_in_continent[['country', 'flag_url']].values.tolist())
    return country, flag_url, '', 0, 0, 'Flag selected!'

def flag_check_guess(user_guess, country, score, errors):
    user_guess_display = user_guess or "[empty]"
    if not user_guess:
        errors += 1
        next_flag = True
        feedback = f'❌ Incorrect! Your guess: {user_guess_display} | Country: {country}'
    elif is_guess_close(user_guess, country):
        score += 1
        highscores["Flag Quiz"] = max(score, highscores["Flag Quiz"])
        next_flag = True
        feedback = f'✅ Correct! Your guess: {user_guess_display} | Country: {country}'
    else:
        errors += 1
        next_flag = True
        feedback = f'❌ Incorrect! Your guess: {user_guess_display} | Correct answer: {country}'
    return score, errors, feedback, '', next_flag

def capitals_start(continent):
    countries_in_continent = countries_df[countries_df['map_reference'] == continent]
    countries_capitals = countries_in_continent[['country', 'capital']].dropna().values.tolist()
    if not countries_capitals:
        return '', '', '', 0, 0, 'No country with a capital found. Please select a different continent.'
    country, capital = random.choice(countries_capitals)
    return country, capital, '', 0, 0, 'Capital selected!'

def capitals_check_guess(user_guess, country, capital, score, errors):
    user_guess_display = user_guess or "[empty]"
    if not user_guess:
        errors += 1
        next_capital = True
        feedback = f'❌ Incorrect! Your guess: {user_guess_display} | Correct answer: {country}'
    elif is_guess_close(user_guess, country):
        score += 1
        highscores["Capital Quiz"] = max(score, highscores["Capital Quiz"])
        next_capital = True
        feedback = f'✅ Correct! Your guess: {user_guess_display} | Country: {country}'
    else:
        errors += 1
        next_capital = True
        feedback = f'❌ Incorrect! Your guess: {user_guess_display} | Correct answer: {country}'
    return score, errors, feedback, '', next_capital

# ---- Gradio UI ----
with gr.Blocks() as demo:
    menu_box = gr.Markdown(visible=True)
    with gr.Row(visible=True) as menu_row:
        btn_hints = gr.Button("Hints Game")
        btn_flag = gr.Button("Flag Quiz")
        btn_capitals = gr.Button("Capital Quiz")
    hints_highscore = gr.State(0)
    flag_highscore = gr.State(0)
    capitals_highscore = gr.State(0)
    with gr.Column(visible=False) as hints_col:
        hints_continent = gr.Dropdown(label='Select Continent', choices=get_continents())
        hints_hint_type = gr.Dropdown(label='Hint Category', choices=['geography', 'history', 'flag', 'food', 'random'])
        hints_country = gr.Textbox(visible=False)
        hints_hint = gr.Markdown()
        hints_user_guess = gr.Textbox(label='Your Guess', interactive=True)
        hints_feedback = gr.Markdown()
        hints_status = gr.Markdown()
        hints_used_hints = gr.State([])
        hints_used_categories = gr.State([])
        hints_hint_nr = gr.State(1)
        hints_score = gr.State(0)
        btn_hints_start = gr.Button("Start New Game")
        btn_hints_hint = gr.Button("Show Next Hint")
        btn_hints_guess = gr.Button("Submit Guess")
        btn_hints_exit = gr.Button("⬅️ Back to Menu")
    with gr.Column(visible=False) as flag_col:
        flag_continent = gr.Dropdown(label='Select Continent', choices=get_continents())
        flag_country = gr.Textbox(visible=False)
        flag_flag_url = gr.Image(label="Flag", image_mode="RGB", height=200, width=300)
        flag_user_guess = gr.Textbox(label='Your Guess', interactive=True)
        flag_feedback = gr.Markdown()
        flag_score = gr.State(0)
        flag_errors = gr.State(0)
        btn_flag_start = gr.Button("Next Flag")
        btn_flag_guess = gr.Button("Submit Guess")
        btn_flag_exit = gr.Button("⬅️ Back to Menu")
    with gr.Column(visible=False) as capitals_col:
        capitals_continent = gr.Dropdown(label='Select Continent', choices=get_continents())
        capitals_country = gr.Textbox(visible=False)
        capitals_capital = gr.Textbox(label="Capital", interactive=False)
        capitals_user_guess = gr.Textbox(label='Your Guess', interactive=True)
        capitals_feedback = gr.Markdown()
        capitals_score = gr.State(0)
        capitals_errors = gr.State(0)
        btn_capitals_start = gr.Button("Next Capital")
        btn_capitals_guess = gr.Button("Submit Guess")
        btn_capitals_exit = gr.Button("⬅️ Back to Menu")
    # Callbacks (wie in deiner letzten Version, kannst du übernehmen)
    # ...

    # SHOW MENU ON LOAD
    def show_menu(hints_score, flag_score, capitals_score):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            f"""
            ## GeoGuesser RAG – Main Menu

            **Highscores:**  
            - Hints Game: {hints_score}
            - Flag Quiz: {flag_score}
            - Capital Quiz: {capitals_score}

            Select a game mode below:
            """
        )
    demo.load(
        show_menu,
        [hints_highscore, flag_highscore, capitals_highscore],
        [menu_row, hints_col, flag_col, capitals_col, menu_box]
    )
    # ... Rest wie gehabt
demo.launch()
