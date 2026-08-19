"""
Autonomous AI Research Agent for AWS Trainium Frontier.

Systematically explores the co-design parameter space (architecture, kernels, optimizer LR, WSD schedules)
to discover Pareto-optimal configurations that minimize validation bits-per-byte (val_bpb)
under the 30-minute Trainium2 single-chip constraint.
"""

import os
import json
import time
import math
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [AutoResearch] %(message)s")
logger = logging.getLogger("AutoResearch")


@dataclass
class ExperimentResult:
    trial_id: str
    model_type: str
    dim: int
    n_layers: int
    n_heads: int
    muon_lr: float
    adamw_lr: float
    val_bpb: float
    val_loss: float
    tokens_per_sec: float
    mfu_percent: float
    training_time_sec: float
    timestamp: float


class AutoResearchAgent:
    """
    Automated Exploration Engine for Trainium Frontier Speedruns.
    """
    def __init__(self, log_dir: str = "./experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history_file = os.path.join(log_dir, "trials_history.json")
        self.trials: List[ExperimentResult] = []
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.trials = [ExperimentResult(**item) for item in data]
                logger.info(f"Loaded {len(self.trials)} past experimental trials.")
            except Exception as e:
                logger.warning(f"Could not load past trial history: {e}")

    def log_trial(self, result: ExperimentResult):
        self.trials.append(result)
        with open(self.history_file, "w") as f:
            json.dump([asdict(t) for t in self.trials], f, indent=2)
        logger.info(f"Logged Trial {result.trial_id}: Val BPB={result.val_bpb:.4f}, Tok/s={result.tokens_per_sec:,.0f}, MFU={result.mfu_percent:.1f}%")

    def get_pareto_frontier(self) -> List[ExperimentResult]:
        """
        Calculates the Pareto frontier minimizing val_bpb while maximizing throughput.
        """
        if not self.trials:
            return []
            
        sorted_trials = sorted(self.trials, key=lambda x: x.val_bpb)
        pareto = []
        max_throughput = -1.0
        
        for t in sorted_trials:
            if t.tokens_per_sec > max_throughput:
                pareto.append(t)
                max_throughput = t.tokens_per_sec
                
        return pareto

    def propose_next_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Heuristic AI-driven search proposing the next optimal configurations to test on Trn2.
        """
        hypotheses = [
            {
                "hypothesis_id": "hyp_01_aspect_ratio_tuning",
                "rationale": "Test deeper vs wider models (16 layers x 512 dim vs 8 layers x 1024 dim) matching 128x128 TensorEngine tiles",
                "model_type": "base",
                "dim": 768,
                "n_layers": 12,
                "muon_lr": 0.025,
                "adamw_lr": 0.0012
            },
            {
                "hypothesis_id": "hyp_02_fine_grained_moe_scaling",
                "rationale": "Activate 8 experts top-2 with SBUF chunked CE to maximize token capacity per FLOP",
                "model_type": "moe",
                "dim": 768,
                "n_layers": 10,
                "muon_lr": 0.02,
                "adamw_lr": 0.001
            },
            {
                "hypothesis_id": "hyp_03_aggressive_muon_lr_wsd",
                "rationale": "Leverage QK-Norm to push Muon peak LR to 0.035 with fast 15% cosine anneal",
                "model_type": "base",
                "dim": 768,
                "n_layers": 12,
                "muon_lr": 0.035,
                "adamw_lr": 0.0015
            }
        ]
        return hypotheses
