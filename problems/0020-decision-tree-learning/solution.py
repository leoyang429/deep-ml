import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    label_counts, entropy, tot = Counter(labels).values(), 0, len(labels)
    for label_count in label_counts:
        p = label_count / tot
        entropy -= p * math.log2(p)
    return entropy

def collect_labels(examples: list[dict], target_attr: str, attr: str = None, value: str = None) -> list:
    labels = []
    for entry in examples:
        if attr is None or entry[attr] == value:
            labels.append(entry[target_attr])
    return labels

def collect_entries(examples: list[dict], attr: str, value: str) -> list:
    entries = []
    for entry in examples:
        if entry[attr] == value:
            entries.append(entry)
    return entries

def collect_values(examples: list[dict], attr: str) -> set:
    values = set()
    for entry in examples:
        values.add(entry[attr])
    return values

def calculate_