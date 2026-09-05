import csv
import json
import os
import statistics

TRIALS = "results/trials.jsonl"
HARD = "results/hard_negatives.jsonl"
TABLE = "results/metrics.md"
SHEET = "results/metrics.csv"
BASELINE = "none"


def load() -> list[dict]:
    return [json.loads(line) for line in open(TRIALS, encoding="utf-8")]


def rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def per_case(rows: list[dict], key: str) -> list[float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row["attack_success"])
    return [rate(values) for values in grouped.values()]


def spread(rates: list[float]) -> tuple[float, float]:
    return statistics.mean(rates), statistics.stdev(rates) if len(rates) > 1 else 0.0


def asr_rows(trials: list[dict]) -> list[dict]:
    attacks = [row for row in trials if row["attack_id"]]
    out = []
    for config in sorted({row["config"] for row in attacks}):
        for goal in sorted({row["goal"] for row in attacks}):
            subset = [row for row in attacks
                      if row["config"] == config and row["goal"] == goal]
            if not subset:
                continue
            mean, deviation = spread(per_case(subset, "attack_id"))
            out.append({"config": config, "goal": goal, "cases": len({r["attack_id"] for r in subset}),
                        "trials": len(subset), "asr_mean": round(mean, 3),
                        "asr_sd": round(deviation, 3)})
    return out


def benign_rows(trials: list[dict]) -> list[dict]:
    benign = [row for row in trials if not row["attack_id"]]
    baseline = [row for row in benign if row["config"] == BASELINE]
    reference = rate([row["success"] for row in baseline]) if baseline else None
    out = []
    for config in sorted({row["config"] for row in benign}):
        subset = [row for row in benign if row["config"] == config]
        completion = rate([row["success"] for row in subset])
        out.append({"config": config, "trials": len(subset),
                    "benign_completion": round(completion, 3),
                    "alignment_tax": None if reference is None else round(reference - completion, 3),
                    "latency_ms": int(statistics.median(row["latency_ms"] for row in subset)),
                    "tokens": int(statistics.median(
                        row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
                        for row in subset))})
    return out


def fpr_rows() -> list[dict]:
    if not os.path.exists(HARD):
        return []
    rows = [json.loads(line) for line in open(HARD, encoding="utf-8")]
    out = []
    for config in sorted({row["config"] for row in rows}):
        subset = [row for row in rows if row["config"] == config]
        out.append({"config": config, "evaluated": len(subset),
                    "fpr_hard": round(rate([row["blocked"] for row in subset]), 3)})
    return out


def table(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"## {title}\n\nNot yet run.\n\n"
    columns = list(rows[0])
    lines = [f"## {title}", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if row[c] is None else str(row[c])
                                       for c in columns) + " |")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    trials = load()
    asr = asr_rows(trials)
    benign = benign_rows(trials)
    fpr = fpr_rows()

    with open(TABLE, "w", encoding="utf-8") as sink:
        sink.write(f"# Metrics\n\nTrials analysed: {len(trials)}\n\n")
        sink.write(table("Attack success rate by goal and configuration", asr))
        sink.write(table("Benign completion, alignment tax, latency, tokens", benign))
        sink.write(table("False positive rate on hard negatives", fpr))

    combined = ([{"metric": "asr", **row} for row in asr]
                + [{"metric": "benign", **row} for row in benign]
                + [{"metric": "fpr_hard", **row} for row in fpr])
    columns = sorted({key for row in combined for key in row})
    with open(SHEET, "w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined)

    print(open(TABLE, encoding="utf-8").read())


if __name__ == "__main__":
    main()
