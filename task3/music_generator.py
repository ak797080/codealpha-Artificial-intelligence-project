"""Train a small LSTM on MIDI files and generate new MIDI sequences."""

import argparse
import random
from pathlib import Path

import music21
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
MIDI_DIR = ROOT / "data" / "midi"
OUTPUT_DIR = ROOT / "output"
CHECKPOINT = OUTPUT_DIR / "music_lstm.pt"
DURATIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def quantize_duration(duration):
    """Map a MIDI quarter-length to the closest supported duration."""
    duration = max(float(duration), 0.125)
    return min(DURATIONS, key=lambda value: abs(value - duration))


def score_to_tokens(score):
    """Extract a note/rest token sequence from the part with most events."""
    parts = list(score.parts)
    candidates = parts or [score]
    selected = max(
        candidates,
        key=lambda part: len(list(part.flatten().notesAndRests)),
    )
    tokens = []

    for event in selected.flatten().notesAndRests:
        duration = quantize_duration(event.duration.quarterLength)
        if isinstance(event, music21.note.Rest):
            tokens.append(f"R:{duration}")
        elif isinstance(event, music21.chord.Chord):
            tokens.append(f"N:{max(pitch.midi for pitch in event.pitches)}:{duration}")
        else:
            tokens.append(f"N:{event.pitch.midi}:{duration}")

    return tokens


def load_midi_sequences(midi_dir):
    """Read all supported MIDI files recursively and return their note tokens."""
    paths = sorted(
        path
        for path in Path(midi_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    sequences = []
    errors = []

    for path in paths:
        try:
            tokens = score_to_tokens(music21.converter.parse(str(path)))
            if len(tokens) >= 2:
                sequences.append((path.name, tokens))
            else:
                errors.append(f"{path.name}: fewer than 2 note/rest events")
        except (OSError, ValueError, music21.exceptions21.Music21Exception) as exc:
            errors.append(f"{path.name}: {exc}")

    if not paths:
        raise ValueError(
            f"No MIDI files found in {midi_dir}. Add .mid or .midi files first."
        )
    if not sequences:
        details = "; ".join(errors[:3])
        raise ValueError(f"No usable MIDI sequences found. {details}")

    return sequences, errors


def make_training_examples(sequences, sequence_length):
    """Create overlapping next-token examples and a vocabulary."""
    vocabulary = sorted({token for _, tokens in sequences for token in tokens})
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    inputs = []
    targets = []
    stride = max(1, sequence_length // 2)

    for _, tokens in sequences:
        encoded = [token_to_id[token] for token in tokens]
        if len(encoded) <= sequence_length:
            continue
        starts = list(range(0, len(encoded) - sequence_length, stride))
        final_start = len(encoded) - sequence_length - 1
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            inputs.append(encoded[start : start + sequence_length])
            targets.append(encoded[start + sequence_length])

    if not inputs:
        raise ValueError(
            "MIDI sequences are too short for the selected sequence length. "
            "Add longer files or reduce --sequence-length."
        )

    return vocabulary, torch.tensor(inputs), torch.tensor(targets)


class MusicLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        output, _ = self.lstm(embedded)
        return self.output(output[:, -1, :])


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sequences, errors = load_midi_sequences(args.data)
    minimum_length = min(len(tokens) for _, tokens in sequences)
    sequence_length = min(args.sequence_length, minimum_length - 1)
    if sequence_length < 1:
        raise ValueError("Each MIDI file must contain at least 2 note/rest events.")

    vocabulary, inputs, targets = make_training_examples(
        sequences, sequence_length
    )
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = MusicLSTM(len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    print(
        f"Training on {len(sequences)} MIDI file(s), {len(vocabulary)} tokens, "
        f"{len(dataset)} examples."
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)
        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vocabulary": vocabulary,
            "sequence_length": sequence_length,
            "embedding_size": 64,
            "hidden_size": 128,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    print(f"Model saved to {args.output}")
    for error in errors:
        print(f"Skipped: {error}")


def sample_next_token(logits, temperature):
    if temperature <= 0:
        raise ValueError("--temperature must be greater than 0.")
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def token_to_event(token):
    parts = token.split(":")
    if parts[0] == "R" and len(parts) == 2:
        return music21.note.Rest(quarterLength=float(parts[1]))
    if parts[0] == "N" and len(parts) == 3:
        return music21.note.Note(
            int(parts[1]), quarterLength=float(parts[2])
        )
    raise ValueError(f"Invalid music token in model: {token}")


def generate(args):
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Train the model first."
        )
    if args.notes < 1:
        raise ValueError("--notes must be at least 1.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0.")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = MusicLSTM(
        len(vocabulary),
        embedding_size=checkpoint["embedding_size"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    seed_length = min(checkpoint["sequence_length"], len(vocabulary))
    seed = [random.randrange(len(vocabulary)) for _ in range(seed_length)]
    generated = []
    with torch.inference_mode():
        for _ in range(args.notes):
            logits = model(torch.tensor([seed]))[0]
            next_token = sample_next_token(logits, args.temperature)
            generated.append(vocabulary[next_token])
            seed.append(next_token)
            seed = seed[-checkpoint["sequence_length"] :]

    melody = music21.stream.Stream()
    melody.append(music21.tempo.MetronomeMark(number=args.tempo))
    for token in generated:
        melody.append(token_to_event(token))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    melody.write("midi", fp=str(args.output))
    print(f"Generated {len(generated)} events and saved MIDI to {args.output}")


def create_demo_data(args):
    """Create a few original short melodies to test the pipeline."""
    args.output.mkdir(parents=True, exist_ok=True)
    patterns = [
        [60, 62, 64, 67, 64, 62, 60, 55],
        [60, 65, 69, 67, 65, 62, 57, 60],
        [67, 69, 71, 72, 71, 69, 67, 62],
        [64, 67, 72, 71, 69, 67, 64, 60],
    ]
    durations = (1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 2.0)

    for melody_index, pattern in enumerate(patterns, start=1):
        melody = music21.stream.Stream()
        melody.append(music21.tempo.MetronomeMark(number=100))
        for repetition in range(4):
            for note_index, pitch in enumerate(pattern):
                adjusted_pitch = pitch + (12 if repetition == 2 else 0)
                melody.append(
                    music21.note.Note(
                        adjusted_pitch,
                        quarterLength=durations[note_index],
                    )
                )
        melody.write("midi", fp=str(args.output / f"demo_{melody_index}.mid"))
    print(f"Created 4 demo MIDI files in {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an LSTM on MIDI files and generate a new melody."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train the LSTM model.")
    train_parser.add_argument("--data", type=Path, default=MIDI_DIR)
    train_parser.add_argument("--output", type=Path, default=CHECKPOINT)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--sequence-length", type=int, default=32)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(run=train)

    generate_parser = commands.add_parser(
        "generate", help="Generate and save a MIDI melody."
    )
    generate_parser.add_argument("--model", type=Path, default=CHECKPOINT)
    generate_parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "generated.mid"
    )
    generate_parser.add_argument("--notes", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--tempo", type=int, default=110)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(run=generate)

    demo_parser = commands.add_parser(
        "make-demo-data", help="Create original sample MIDI files for testing."
    )
    demo_parser.add_argument("--output", type=Path, default=MIDI_DIR)
    demo_parser.set_defaults(run=create_demo_data)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "seed"):
        random.seed(args.seed)
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
