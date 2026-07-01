"""
IA/model.py — Classe Model : wrapper d'inférence pour un modèle chargé/sauvegardé.

Un Model est une structure de données contenant :
  - model_type : 'cnn' | 'rnn' | 'transformer' | 'gan' | 'ldm' | 'slm' | 'image_classifier'
  - config     : dict de configuration (hyperparams, shapes, vocab...)
  - weights    : dict {nom: ndarray} des poids
  - predict_fn : fonction d'inférence injectée par le Trainer lors de l'entraînement

Usage typique :
    trainer = Trainer()
    model = trainer.train(type='rnn', epochs=500)
    out = model.predict(X_new)
    model.save('mon_modele.gy')
    model2 = Model.load('mon_modele.gy')
"""
import os
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import ia_format


# Registre des fonctions d'inférence par type de modèle.
# Chaque fonction doit avoir la signature : predict_fn(weights, config, X) -> y_pred
# Le Trainer enregistre la fonction pendant l'entraînement.
_PREDICT_REGISTRY: Dict[str, Callable] = {}


def register_predict_fn(model_type: str):
    """Décorateur pour enregistrer une fonction d'inférence."""
    def deco(fn):
        _PREDICT_REGISTRY[model_type] = fn
        return fn
    return deco


class Model:
    """Modèle entraîné, capable d'inférence et de persistance .gy."""

    def __init__(self, model_type: str, config: Dict[str, Any],
                 weights: Dict[str, np.ndarray],
                 predict_fn: Optional[Callable] = None):
        self.model_type = model_type
        self.config = config
        self.weights = weights
        self.predict_fn = predict_fn or _PREDICT_REGISTRY.get(model_type)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def predict(self, X) -> Any:
        """Effectue une prédiction sur X.

        Args:
            X: entrée (ndarray, ou liste d'ndarray selon le type de modèle).

        Returns:
            Prédiction (shape et type dépendent du modèle).
        """
        if self.predict_fn is None:
            raise RuntimeError(
                f"Aucune fonction d'inférence enregistrée pour le type "
                f"'{self.model_type}'. Chargez le modèle via Trainer.load() "
                f"ou enregistrez une fonction via register_predict_fn()."
            )
        return self.predict_fn(self.weights, self.config, X)

    # ------------------------------------------------------------------
    # Persistance .gy (binaire safe)
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Sauvegarde le modèle au format .gy (binaire safe).

        Args:
            path: chemin du fichier (extension .gy automatique si autre extension).

        Returns:
            Le chemin absolu du fichier sauvegardé.
        """
        # Forcer l'extension .gy
        base, ext = os.path.splitext(path)
        if ext and ext != '.gy':
            path = base + '.gy'
        elif not ext:
            path = path + '.gy'
        # Si juste un nom sans répertoire, sauvegarder dans models/
        if not os.path.dirname(path):
            from .config import MODELS_DIR
            path = os.path.join(MODELS_DIR, path)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        header = {
            'type': self.model_type,
            'config': self.config,
            'weights_meta': {k: list(v.shape) for k, v in self.weights.items()},
        }
        ia_format.save_model(path, header, self.weights)
        return path

    @classmethod
    def load(cls, path: str) -> 'Model':
        """Charge un modèle depuis un fichier .gy.

        Args:
            path: chemin du fichier .gy.

        Returns:
            Instance de Model avec predict_fn récupérée depuis le registre.
        """
        header, weights = ia_format.load_model(path)
        model_type = header.get('type', 'unknown')
        config = header.get('config', {})
        # La predict_fn est récupérée depuis le registre ; si le type n'est
        # pas enregistré, predict() lèvera une RuntimeError explicite.
        return cls(model_type, config, weights)

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Retourne un résumé textuel du modèle."""
        lines = [
            f"Model(type={self.model_type!r})",
            f"  weights: {len(self.weights)} tensors",
        ]
        for name, w in self.weights.items():
            lines.append(f"    - {name}: shape={w.shape}, dtype={w.dtype}")
        if self.config:
            lines.append(f"  config:")
            for k, v in self.config.items():
                v_repr = repr(v)
                if len(v_repr) > 80:
                    v_repr = v_repr[:77] + '...'
                lines.append(f"    - {k}: {v_repr}")
        if self.predict_fn is None:
            lines.append("  predict_fn: NON ENREGISTRÉE (predict() lèvera une erreur)")
        else:
            lines.append("  predict_fn: enregistrée ✓")
        return '\n'.join(lines)

    def __repr__(self):
        return f"Model(type={self.model_type!r}, weights={len(self.weights)})"
