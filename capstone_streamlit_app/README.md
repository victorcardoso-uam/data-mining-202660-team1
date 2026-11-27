# PrestigePick · Viewer Persona Intelligence Engine (Team 1, Milestone 3/3)

Streamlit dashboard for the HBO Max UX & Content Team. Movies are clustered with
K-Means (k = 6) into viewer personas; Decision Tree, Logistic Regression and a
Neural Network (MLP) are trained to assign new titles to a persona. The
**Award Caliber** carousel takes titles whose probability of the
"Prestige Awards Chaser" persona is above the cutoff in the sidebar.

## Run

```bash
cd capstone_streamlit_app
python3 -m pip install -r requirements.txt
python3 train_models.py      # builds models/*.joblib + feature_importance.json
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your LLM API key
streamlit run app.py
```

## Layout

```
capstone_streamlit_app/
├── app.py                 # 4 tabs + AI Assistant tab, loads the .joblib models (no training)
├── app_session24_demo.py  # Activity 20 Session 24 template, kept for grading (NOT part of the final app)
├── llm_assistant.py       # LLM chatbot grounded in the app's data, metrics and live What-If decision
├── pipeline.py            # shared cleaning + feature engineering
├── train_models.py        # trains and saves the models + benchmark metrics
├── data/                  # HBO_MAX_Content_extended.csv, People_Dataset.csv
├── models/                # kmeans, scaler, decision tree, logistic, ANN (.joblib) + metrics
├── .streamlit/config.toml # dark theme
├── .streamlit/secrets.toml.example  # LLM provider settings (copy to secrets.toml, never commit it)
└── requirements.txt
```

The features and K-Means settings are the same as `src/movie_viewer_clusters.py`,
so the six clusters match the Milestone 2 personas exactly.

## AI Assistant (LLM chatbot)

Tab 5 answers questions about data patterns, feature importance rankings and the live
What-If decision. On every question the app sends the LLM a fact sheet built from its own
data (persona profiles, benchmark metrics, permutation importance, Logistic Regression
coefficients, the current What-If inputs and probabilities, and any catalog movie named in the
question). The model is told to answer only from those facts; the expander under the chat shows
exactly what it sees.

Providers: Claude (Anthropic) or any OpenAI-compatible API (OpenAI, Google Gemini, Groq,
OpenRouter, local Ollama). Settings go in `.streamlit/secrets.toml` (see the `.example` file).
`secrets.toml` is git-ignored, so API keys never reach GitHub.

## Reliability and safety of the AI Assistant

| Risk | Protection |
|---|---|
| Prompt injection, role play, requests to reveal the prompt or keys | Guardrail rules in the system prompt; tool outputs and user text are treated as data, not instructions; API keys never reach the model |
| Off-topic questions | The assistant only answers about PrestigePick |
| False premises ("why did the ANN win?") | The assistant corrects the premise from the facts first |
| Invented numbers | Every number in an answer is checked against the app's data; the result is shown under the answer (✅ verified / ⚠️ double-check) |
| Impossible inputs (IMDb 15, year 3000) | The classifier tool refuses to run |
| Inputs outside the training data | Tab 4 and the assistant warn that the prediction is an extrapolation and recommend human review |
| Busy or slow LLM provider | Retries, backup models, 60 s limit; movie questions are still answered straight from the data |
| Very long inputs | Questions are capped at 800 characters; temperature 0 for repeatable answers |

### Tests

```bash
cd capstone_streamlit_app
python3 -m pip install pytest
python3 -m pytest tests -q          # 37 offline tests: tools, validation, fact-check (no key needed)
python3 tests/red_team.py           # 15 adversarial questions against the real LLM -> tests/red_team_report.md
```
