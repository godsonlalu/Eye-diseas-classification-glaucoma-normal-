# OcuLens Streamlit app

## Run

From the workspace root:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses `mobilenetv2_trained_model.keras`, with the same `224x224` resize, `[0, 1]` normalization, and `0.40` glaucoma threshold as the training notebook.

## LLM explanations

For local explanations, install and start Ollama, then pull a model such as `llama3.2:3b`:

```powershell
ollama pull llama3.2:3b
ollama serve
```

For Hugging Face, set `HF_TOKEN` in the environment or add it to `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_..."
```

The language model receives the classifier result and probability, not the image. Its text is educational context, not visual evidence or a diagnosis.