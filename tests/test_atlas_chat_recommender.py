import json
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from causal_algorithms_atlas import chat_recommender, rag_chat
from causal_algorithms_atlas.dataset_profile import profile_dataset


def _profile():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    return profile_dataset(frame)


def _consistent_response(profile):
    """Resposta valida do 'modelo': percentuais reais do perfil, nenhuma premissa exigida."""
    return json.dumps(
        {
            "stationary_fraction_pct": round(profile.stationary_fraction * 100),
            "dataset_is_majority_stationary": profile.mostly_stationary,
            "linear_fraction_pct": round(profile.linear_fraction * 100),
            "dataset_is_majority_linear": profile.mostly_linear,
            "non_gaussian_fraction_pct": round(profile.non_gaussian_fraction * 100),
            "dataset_is_majority_non_gaussian": profile.mostly_non_gaussian,
            "algorithm_requires_stationarity": False,
            "algorithm_requires_linearity": False,
            "algorithm_requires_non_gaussian_errors": False,
            "include": True,
            "reason": "nenhuma premissa violada",
        }
    )


class ChatRecommenderParallelTests(unittest.TestCase):
    def test_parallel_matches_sequential_and_preserves_order(self):
        profile = _profile()
        response = _consistent_response(profile)
        with mock.patch.object(rag_chat, "call_ollama", return_value=response):
            sequential = chat_recommender.recommend_methods_via_chat(profile)
            parallel = chat_recommender.recommend_methods_via_chat(profile, max_workers=4)
        self.assertEqual([d.name for d in sequential.decisions], [d.name for d in parallel.decisions])
        self.assertEqual(sequential.included, parallel.included)
        self.assertEqual(sequential.justification, parallel.justification)
        self.assertGreater(len(parallel.decisions), 1)

    def test_parallel_runs_calls_concurrently(self):
        profile = _profile()
        response = _consistent_response(profile)
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        def slow_call(*_args, **_kwargs):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return response

        with mock.patch.object(rag_chat, "call_ollama", side_effect=slow_call):
            chat_recommender.recommend_methods_via_chat(profile, max_workers=4)
        self.assertGreater(state["peak"], 1)

        state["peak"] = 0
        with mock.patch.object(rag_chat, "call_ollama", side_effect=slow_call):
            chat_recommender.recommend_methods_via_chat(profile)  # padrao: sequencial
        self.assertEqual(state["peak"], 1)


if __name__ == "__main__":
    unittest.main()
