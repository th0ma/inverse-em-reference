from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SelectionState:
    best_macro_f1: float = -math.inf
    best_epoch: int = 0
    best_update: int = 0
    early_reference: float = -math.inf
    bad_epochs: int = 0

    def update(self, macro_f1: float, epoch: int, updates: int, *, threshold: float = 1e-4, patience: int = 20):
        value = float(macro_f1); best_improved = value > self.best_macro_f1
        if best_improved:
            self.best_macro_f1 = value; self.best_epoch = int(epoch); self.best_update = int(updates)
        if self.early_reference == -math.inf or value > self.early_reference + threshold:
            self.early_reference = value; self.bad_epochs = 0
        else: self.bad_epochs += 1
        return best_improved, self.bad_epochs >= patience
