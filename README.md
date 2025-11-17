# Baby LLM

A tiny "baby" chatbot that picks up language as you talk to it. Every user
message or fed document updates a lightweight n-gram model so the bot slowly
learns new words and transitions over time.

## How it works

- **Incremental learning** - `BabyLanguageModel` tracks unigram, bigram, and
  trigram counts so it remembers short word sequences. Training happens every
  time you chat or feed it text.
- **Rich embeddings** - the `BabyEmbeddings` component uses distance-weighted
  co-occurrences, adaptive learning rates, and negative sampling to learn
  distributed representations on the fly. You can even bootstrap it with
  pre-trained GloVe files.
- **Neural RNN head (PyTorch)** - a lightweight GRU + multi-head attention
  block (built with PyTorch) predicts the next embedding vector and biases
  sampling toward more coherent continuations.
- **Simple generation** - responses sample words based on these statistics and
  embedding affinity, preferring trigram matches first, so babbling becomes
  more coherent as the model grows. The tokenizer keeps punctuation tokens, so
  the baby can mimic commas, questions, and exclamations it has observed.
- **Replay & logging** - every training snippet lands in a replay buffer and a
  `training_log.jsonl` file so you can retrain the neural head or export/import
  weights later.
- **Persistence** - brains are saved as JSON so you can pause and resume the
  baby's education whenever you like.

## Getting started

Install dependencies (PDF support relies on `pypdf`):

```bash
pip install -r requirements.txt
```

> **GPU note:** The default `pip install torch` pulls the CPU build. Follow
> [PyTorch's install guide](https://pytorch.org/get-started/locally/) and use
> their CUDA wheel URL (e.g., `pip install --index-url https://download.pytorch.org/whl/cu124 torch`)
> if you want the RNN head to run on your GPU.

Then launch the REPL:

```bash
python main.py
```

Useful CLI arguments:

- `--model-path baby.json` - choose where the brain file lives (defaults to
  `baby_model.json`).
- `--max-response-words 60` - cap the babble length.
- `--seed 123` - make the randomness deterministic for demos/tests.
- `--embedding-dim 128` - choose the embedding size for new brains.
- `--device cuda` - force the PyTorch RNN head onto a specific device (`auto`
  picks CUDA when available, otherwise CPU).

## Chat commands

Inside the REPL you can use the following slash commands:

- `/help` - list all commands.
- `/feed <file>` - read and learn from UTF-8 text or PDF files.
- `/scrape <folder>` - recursively ingest every supported document (text/PDF)
  inside the folder. Hitting `Ctrl+C` mid-scrape stops gracefully and
  auto-saves whatever was learned so far.
- `/web <url> [max_chars]` - fetch a web page, clean it with BeautifulSoup, and
  feed the extracted text to the baby.
- `/crawl <url> [max_pages max_depth max_chars infinite]` - ethically crawl a
  domain (respecting `robots.txt`, staying on-site). Include `infinite` to keep
  crawling until you hit `Ctrl+C` (progress auto-saves on interrupt).
- `/import-emb <file> [limit]` - load pre-trained embeddings in GloVe format
  (optionally limiting how many rows are ingested).
- `/similar <word>` - inspect the top embedding neighbors the baby knows.
- `/feedback +/-text` - give positive/negative reinforcement to nudge the baby.
- `/replay [rounds]` - re-run the replay buffer through the neural head for
  extra fine-tuning.
- `/export-head <file>` / `/import-head <file>` - save or restore the PyTorch
  head state.
- `/retrain-log [reset]` - replay everything in `training_log.jsonl`
  (`reset` optionally refreshes the head first).
- `/save [file]` - persist the brain (defaults to the current model path).
- `/load <file>` - swap to a different saved brain.
- `/stats` - show vocabulary, utterances, transitions, and embedding counts.
- `/reset` - wipe everything and start over.
- `/quit` or `/exit` - leave the chat (auto-saves on exit).

> Tip: after upgrading, run `/retrain-log reset` to rebuild the neural head
> from your `training_log.jsonl`. If you created your brain before logging was
> enabled, re-run `/scrape <folder>` on your historical documents to capture
> them under the new training standards.

## Project structure

- `baby_llm/tokenizer.py` - lower-case tokenizer shared by the project.
- `baby_llm/embeddings.py` - maintains the online embeddings (weighted updates,
  negative sampling, optional pre-trained import).
- `baby_llm/rnn_head.py` - PyTorch GRU head that predicts the next embedding
  vector to bias generation.
- `baby_llm/memory.py` - lightweight replay buffer used for head fine-tuning.
- `baby_llm/documents.py` - document ingestion helpers (text/PDF).
- `baby_llm/model.py` - the incremental n-gram (uni/bi/tri) learner plus
  embeddings, persistence, and stats.
- `main.py` - interactive CLI that wires everything together.

Feel free to extend the model with richer features (e.g., trigrams, RNN heads,
or better sampling) while preserving the same friendly "baby" UX.
