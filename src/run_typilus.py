#!/bin/python
import os
from glob import iglob
import json
import sys
from tempfile import TemporaryDirectory
from typing import Sequence, Tuple, List

import requests
from dpu_utils.utils import load_jsonl_gz
from ptgnn.implementations.typilus.graph2class import Graph2Class

from changeutils import get_changed_files
from annotationutils import (
    annotate_line,
    find_annotation_line,
    group_suggestions,
    annotation_rewrite,
)
from graph_generator.extract_graphs import extract_graphs, Monitoring
from pathlib import Path

work_dir = os.path.dirname(__file__)


class TypeSuggestion:
    def __init__(
        self,
        filepath: str,
        name: str,
        file_location: Tuple[int, int],
        suggestion: str,
        symbol_kind: str,
        confidence: float,
        annotation_lineno: int = 0,
        is_disagreement: bool = False,
    ):
        self.filepath = filepath
        self.name = name
        self.file_location = file_location
        self.suggestion = suggestion
        self.symbol_kind = symbol_kind
        self.confidence = confidence
        self.annotation_lineno = annotation_lineno
        self.is_disagreement = is_disagreement

    def __repr__(self) -> str:
        return (
            f"Suggestion@{self.filepath}:{self.file_location} "
            f"Symbol Name: `{self.name}` Suggestion `{self.suggestion}` "
            f"Confidence: {self.confidence:.2%} "
            f"Disagreement?: {self.is_disagreement}"
        )


def run_model(repo_path, out_dir, debug: bool = False):
    typing_rules_path = os.path.join(work_dir, "metadata", "typingRules.json")
    files_to_extract = {str(f.relative_to(repo_path)) for f in Path(repo_path).glob("**/*.py")}
    if debug:
        print("Files to extract:", files_to_extract)
    extract_graphs(
        repo_path,
        typing_rules_path,
        files_to_extract,
        target_folder=out_dir,
    )

    def data_iter():
        for datafile_path in iglob(os.path.join(out_dir, "*.jsonl.gz")):
            print(f"Looking into {datafile_path}...")
            for graph in load_jsonl_gz(datafile_path):
                yield graph

    model_path = Path(os.path.join(work_dir, "model.pkl.gz"))
    model, nn = Graph2Class.restore_model(model_path, "cpu")

    type_suggestions: List[TypeSuggestion] = []
    for graph, predictions in model.predict(data_iter(), nn, "cpu"):
        # predictions has the type: Dict[int, Tuple[str, float]]
        filepath = graph["filename"]

        if debug:
            print("Predictions:", predictions)
            print("SuperNodes:", graph["supernodes"])

        for supernode_idx, (predicted_type, predicted_prob) in predictions.items():
            supernode_data = graph["supernodes"][str(supernode_idx)]
            # if supernode_data["type"] == "variable":
            #     continue  # Do not suggest annotations on variables for now.
            lineno, colno = supernode_data["location"]
            suggestion = TypeSuggestion(
                filepath,
                supernode_data["name"],
                (lineno, colno),
                annotation_rewrite(predicted_type),
                supernode_data["type"],
                predicted_prob,
                is_disagreement=supernode_data["annotation"] != "??"
                and supernode_data["annotation"] != predicted_type,
            )

            type_suggestions.append(suggestion)

    return type_suggestions


def suggestions_to_json(suggestions: Sequence[TypeSuggestion]):
    file2suggestions = dict()
    for s in suggestions:
        s_list = file2suggestions.setdefault(s.filepath, [])
        s_list.append(
            {
                "location": s.file_location,
                "pred": s.suggestion,
            }
        )
    return file2suggestions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _, in_dir, out_dir = sys.argv
    else:
        in_dir = os.path.join(work_dir, "..", "data")
        out_dir = os.path.join(work_dir, "..", "data_out")
    if not Path(in_dir).exists():
        print(f"Input directory: {in_dir} does not exist.")
        sys.exit(1)
    if not Path(out_dir).exists():
        print(f"Output directory: {out_dir} does not exist.")
        sys.exit(1)
    preds = run_model(in_dir, out_dir, debug=False)
    out_json = suggestions_to_json(preds)
    (Path(out_dir) / "predictions.json").write_text(json.dumps(out_json, indent=2))
    for p in preds:
        print(p)
