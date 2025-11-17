"""Interactive CLI for the baby language-learning chatbot."""

from __future__ import annotations

import argparse
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Tuple

from baby_llm import BabyLanguageModel
from baby_llm.documents import DocumentError, collect_documents, read_document
from baby_llm.web import WebScrapeError, crawl_domain, fetch_url_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a tiny language model that learns from you."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("baby_model.json"),
        help="Where to load/save the baby's brain (JSON).",
    )
    parser.add_argument(
        "--max-response-words",
        type=int,
        default=40,
        help="Maximum number of words in a generated reply.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed to make the chatbot deterministic.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding vector size for brand-new brains (ignored when loading).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Force the PyTorch head onto a device (default: auto-detect).",
    )
    return parser.parse_args()


def resolve_device_choice(choice: str | None) -> Optional[str]:
    if not choice or choice == "auto":
        return None
    return choice


def load_or_create_model(
    path: Path, seed: int | None, embedding_dim: int, device: Optional[str]
) -> BabyLanguageModel:
    if path.exists():
        try:
            model = BabyLanguageModel.load(path, rng_seed=seed, device=device)
            print(f"Loaded existing brain from {path}")
            return model
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load {path}: {exc}")
            sys.exit(1)
    print("Starting with a brand-new baby brain.")
    return BabyLanguageModel(
        rng_seed=seed, embedding_dim=embedding_dim, device=device
    )


def handle_command(
    command_line: str,
    *,
    model: BabyLanguageModel,
    model_path: Path,
) -> Tuple[BabyLanguageModel, Path, bool]:
    """Execute slash commands. Returns (model, path, should_exit)."""
    name, _, argument = command_line.strip().partition(" ")
    name = name.lower()
    argument = argument.strip()
    if name in {"/quit", "/exit"}:
        return model, model_path, True
    if name == "/help":
        print(
            "Commands: /help /feed <file> /web <url> [max_chars] /crawl <url> [pages depth chars] /scrape <folder> /import-emb <file> [limit] "
            "/save [file] /load <file> "
            "/stats /similar <word> /feedback +/-text /replay [n] "
            "/export-head <file> /import-head <file> /retrain-log [reset] /reset /quit"
        )
        return model, model_path, False
    if name == "/feed":
        if not argument:
            print("Provide a file path after /feed.")
            return model, model_path, False
        file_path = Path(argument)
        if not file_path.exists():
            print(f"Cannot find {file_path}")
            return model, model_path, False
        try:
            text = read_document(file_path)
        except DocumentError as exc:
            print(f"Could not read {file_path}: {exc}")
            return model, model_path, False
        except Exception as exc:  # pragma: no cover - unexpected
            print(f"Unexpected error while reading {file_path}: {exc}")
            return model, model_path, False
        learned = model.train_on_text(text)
        print(f"Ate {learned} tokens from {file_path}")
        return model, model_path, False
    if name == "/web":
        if not argument:
            print("Usage: /web <url> [max_chars]")
            return model, model_path, False
        url, _, maybe_limit = argument.partition(" ")
        max_chars: Optional[int] = None
        if maybe_limit:
            try:
                max_chars = max(500, int(maybe_limit))
            except ValueError:
                print("max_chars must be an integer.")
                return model, model_path, False
        try:
            result = fetch_url_text(url, max_chars=max_chars or 20000)
        except WebScrapeError as exc:
            print(f"Could not scrape {url}: {exc}")
            return model, model_path, False
        learned = model.train_on_text(result.text)
        title = f" '{result.title}'" if result.title else ""
        print(
            f"Ate {learned} tokens from{title} ({result.characters} chars, {result.url})"
        )
        return model, model_path, False
    if name == "/crawl":
        args = argument.split()
        if not args:
            print(
                "Usage: /crawl <url> [max_pages=5] [max_depth=1] [max_chars=20000] [infinite=yes/no]"
            )
            return model, model_path, False
        url = args[0]
        max_pages = 5
        max_depth = 1
        max_chars = 20000
        infinite = False
        if len(args) > 1:
            if args[1].lower() == "infinite":
                infinite = True
            else:
                try:
                    max_pages = max(1, int(args[1]))
                except ValueError:
                    print("max_pages must be an integer.")
                    return model, model_path, False
        if len(args) > 2:
            if args[2].lower() == "infinite":
                infinite = True
            else:
                try:
                    max_depth = max(1, int(args[2]))
                except ValueError:
                    print("max_depth must be an integer.")
                    return model, model_path, False
        if len(args) > 3:
            if args[3].lower() in {"infinite", "yes"}:
                infinite = True
            else:
                try:
                    max_chars = max(500, int(args[3]))
                except ValueError:
                    print("max_chars must be an integer.")
                    return model, model_path, False
        if len(args) > 4:
            if args[4].lower() in {"infinite", "yes"}:
                infinite = True
        total_tokens = 0
        processed = 0
        try:
            while True:
                pages = crawl_domain(
                    url,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    max_chars=max_chars,
                    infinite=infinite,
                )
                for page in pages:
                    learned = model.train_on_text(page.text)
                    total_tokens += learned
                    processed += 1
                    title = f" '{page.title}'" if page.title else ""
                    print(
                        f"[{processed}] Ate {learned} tokens from{title} ({page.url})"
                    )
                if not infinite:
                    break
        except KeyboardInterrupt:
            print(
                f"\nCrawl interrupted after {processed} page(s). "
                f"Tokens absorbed: {total_tokens}"
            )
            model.save(model_path)
            print(f"Progress saved to {model_path}")
            return model, model_path, False
        except WebScrapeError as exc:
            print(f"Crawler stopped: {exc}")
            return model, model_path, False
        print(f"Crawl complete. Total tokens absorbed: {total_tokens}")
        return model, model_path, False
    if name == "/feedback":
        if not argument or argument[0] not in "+-":
            print("Usage: /feedback +I liked this or /feedback -This was confusing")
            return model, model_path, False
        positive = argument[0] == "+"
        payload = argument[1:].strip()
        if not payload:
            print("Provide feedback text after +/-")
            return model, model_path, False
        model.apply_feedback(payload, positive)
        print("Thanks for the feedback! Logged.")
        return model, model_path, False
    if name == "/scrape":
        if not argument:
            print("Usage: /scrape <folder>")
            return model, model_path, False
        folder = Path(argument)
        try:
            documents = collect_documents(folder)
        except (FileNotFoundError, DocumentError) as exc:
            print(f"Cannot scrape: {exc}")
            return model, model_path, False
        total_tokens = 0
        processed_files = 0
        try:
            for idx, document in enumerate(documents, start=1):
                try:
                    text = read_document(document)
                except DocumentError as exc:
                    print(f"[{idx}/{len(documents)}] Skipped {document}: {exc}")
                    continue
                except Exception as exc:  # pragma: no cover
                    print(f"[{idx}/{len(documents)}] Unexpected error {document}: {exc}")
                    continue
                learned = model.train_on_text(text)
                total_tokens += learned
                processed_files = idx
                print(f"[{idx}/{len(documents)}] Ate {learned} tokens from {document}")
        except KeyboardInterrupt:
            print(
                f"\nScrape interrupted after {processed_files} file(s). "
                f"Tokens absorbed so far: {total_tokens}"
            )
            model.save(model_path)
            print(f"Progress saved to {model_path}")
            return model, model_path, False
        print(f"Scrape complete. Total tokens absorbed: {total_tokens}")
        return model, model_path, False
    if name == "/replay":
        rounds = 1
        if argument:
            try:
                rounds = max(1, int(argument))
            except ValueError:
                print("Rounds must be an integer.")
                return model, model_path, False
        sequences = model.replay_memory(rounds=rounds)
        print(f"Replayed {sequences} memory sequence(s).")
        return model, model_path, False
    if name in {"/import-emb", "/import-embeddings"}:
        if not argument:
            print("Usage: /import-emb <glove.txt> [limit]")
            return model, model_path, False
        file_arg, _, limit_arg = argument.partition(" ")
        file_path = Path(file_arg)
        if not file_path.exists():
            print(f"Cannot find {file_path}")
            return model, model_path, False
        limit_value: Optional[int] = None
        if limit_arg:
            try:
                limit_value = int(limit_arg)
            except ValueError:
                print("Limit must be an integer.")
                return model, model_path, False
        try:
            loaded = model.import_embeddings(file_path, limit=limit_value)
        except Exception as exc:
            print(f"Could not import embeddings: {exc}")
            return model, model_path, False
        print(f"Imported {loaded} embeddings from {file_path}")
        return model, model_path, False
    if name == "/similar":
        if not argument:
            print("Provide a seed word after /similar.")
            return model, model_path, False
        neighbors = model.similar_words(argument.lower())
        if not neighbors:
            print(
                f"I don't know enough about '{argument}'. "
                "Try feeding me more documents that include it."
            )
        else:
            formatted = ", ".join(f"{word}:{score:.2f}" for word, score in neighbors)
            print(f"Words like '{argument}': {formatted}")
        return model, model_path, False
    if name == "/save":
        dest = Path(argument) if argument else model_path
        model.save(dest)
        print(f"Saved brain to {dest}")
        return model, dest, False
    if name == "/load":
        if not argument:
            print("Provide a model path after /load.")
            return model, model_path, False
        new_path = Path(argument)
        if not new_path.exists():
            print(f"No saved brain at {new_path}")
            return model, model_path, False
        try:
            model = BabyLanguageModel.load(new_path, device=model.device_preference)
        except UnicodeDecodeError:
            print(
                f"{new_path} does not look like a saved brain. "
                "Use /feed <file> to teach from documents such as PDFs."
            )
            return model, model_path, False
        except JSONDecodeError:
            print(
                f"{new_path} is not valid JSON. "
                "Did you mean to feed a document instead? Try /feed <file>."
            )
            return model, model_path, False
        except Exception as exc:
            print(f"Failed to load {new_path}: {exc}")
            return model, model_path, False
        print(f"Loaded new brain from {new_path}")
        return model, new_path, False
    if name == "/stats":
        stats = model.stats()
        print(
            f"Tokens: {stats.tokens_seen} | Utterances: {stats.utterances_seen} | "
            f"Vocab: {stats.vocabulary_size} | "
            f"Bigrams: {stats.bigram_transitions} | "
            f"Trigrams: {stats.trigram_transitions} | "
            f"Embeddings: {stats.embedding_terms} | "
            f"RNN: {stats.head_device}"
        )
        return model, model_path, False
    if name == "/export-head":
        if not argument:
            print("Usage: /export-head <file>")
            return model, model_path, False
        try:
            model.export_head(argument)
        except Exception as exc:
            print(f"Failed to export head: {exc}")
            return model, model_path, False
        print(f"Exported head to {argument}")
        return model, model_path, False
    if name == "/import-head":
        if not argument:
            print("Usage: /import-head <file>")
            return model, model_path, False
        head_path = Path(argument)
        if not head_path.exists():
            print(f"No file at {head_path}")
            return model, model_path, False
        try:
            model.import_head(head_path)
        except Exception as exc:
            print(f"Failed to import head: {exc}")
            return model, model_path, False
        print(f"Imported head from {head_path}")
        return model, model_path, False
    if name == "/retrain-log":
        reset = argument.strip().lower() == "reset"
        count = model.retrain_from_log(reset_head=reset)
        if count == 0:
            print("No logged entries found.")
        else:
            print(f"Re-trained head on {count} logged entry(ies).")
        return model, model_path, False
    if name == "/reset":
        model = BabyLanguageModel(
            label=model.label,
            embedding_dim=model.embeddings.dim,
            device=model.device_preference,
        )
        print("Baby brain reset. Start teaching again!")
        return model, model_path, False

    print("Unknown command. Try /help.")
    return model, model_path, False


def run_chat(
    model: BabyLanguageModel,
    *,
    model_path: Path,
    max_response: int,
) -> Tuple[BabyLanguageModel, Path]:
    print("Type messages to teach the baby. Use /help for commands.")
    while True:
        try:
            message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue

        if message.startswith("/"):
            model, model_path, should_exit = handle_command(
                message, model=model, model_path=model_path
            )
            if should_exit:
                break
            continue

        learned = model.train_on_text(message)
        response = model.reply(message, max_words=max_response)
        print(f"{model.label}> {response} (absorbed {learned} words)")

    return model, model_path


def main() -> None:
    args = parse_args()
    device = resolve_device_choice(args.device)
    model = load_or_create_model(
        args.model_path, args.seed, args.embedding_dim, device
    )
    model, path = run_chat(model, model_path=args.model_path, max_response=args.max_response_words)
    model.save(path)
    print(f"Saved the baby's brain to {path}")


if __name__ == "__main__":
    main()
