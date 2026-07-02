"""
quick_test_custom_data.py — Test rapide avec de vrais X/y minimaux (pas les
données démo générées automatiquement par la lib).

But : vérifier que chaque famille de modèle accepte bien des données
personnalisées, avec le format attendu, sur un dataset volontairement tout
petit (quelques échantillons, 1-2 epochs) pour que ça tourne en quelques
secondes.

Usage :
    Placer ce script au même niveau que le dossier IA/, puis :
        python3 quick_test_custom_data.py

Note : train_cnn_nd, train_gan_1d/nd/3d et train_ldm_image/audio ne prennent
pas de X/y en paramètre (ce sont des modèles génératifs ou à volume généré
en interne) — ils sont testés avec juste très peu d'epochs pour rester
rapides.
"""

import os
import sys
import traceback

IA_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IA_PATH)

os.makedirs('/tmp/gy_models_quick', exist_ok=True)

import numpy as np
import IA

results = {}


def run(name, fn):
    try:
        r = fn()
        results[name] = ("OK", None)
        print(f"[OK] {name}")
        return r
    except Exception as e:
        results[name] = ("FAIL", f"{type(e).__name__}: {e}")
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# MLP — X: (n_samples, n_features), y: (n_samples, 1)
# ============================================================================
X_mlp = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.9, 0.8, 0.7, 0.6],
    [0.2, 0.1, 0.4, 0.3],
    [0.8, 0.9, 0.6, 0.7],
    [0.0, 0.0, 0.1, 0.1],
    [1.0, 1.0, 0.9, 0.9],
])
y_mlp = np.array([[0], [1], [0], [1], [0], [1]])

run("mlp", lambda: IA.train_mlp(
    X=X_mlp, y=y_mlp, epochs=2,
    save_path='/tmp/gy_models_quick/mlp.gy'))

# ============================================================================
# RNN — X: (batch, seq_len, input_size), y: (batch, 1)
# ============================================================================
X_rnn = np.array([
    [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
    [[0.9, 0.8], [0.8, 0.7], [0.7, 0.6]],
    [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]],
    [[1.0, 0.9], [0.9, 0.8], [0.8, 0.7]],
])
y_rnn = np.array([[0], [1], [0], [1]])

run("rnn", lambda: IA.train_rnn(
    X=X_rnn, y=y_rnn, epochs=2,
    save_path='/tmp/gy_models_quick/rnn.gy'))

# ============================================================================
# CNN2D — X: (n_samples, H, W) avec H,W = input_shape (5,5 par défaut)
# ============================================================================
X_cnn2d = np.array([
    np.eye(5),
    np.rot90(np.eye(5)),
    np.ones((5, 5)),
    np.zeros((5, 5)),
])
y_cnn2d = np.array([[1], [1], [0], [0]])

run("cnn2d", lambda: IA.train_cnn2d(
    X=X_cnn2d, y=y_cnn2d, epochs=2,
    save_path='/tmp/gy_models_quick/cnn2d.gy'))

# ============================================================================
# CNN N-D — pas de X/y personnalisable, juste peu d'epochs
# ============================================================================
run("cnn_nd", lambda: IA.train_cnn_nd(
    epochs=2, save_path='/tmp/gy_models_quick/cnn_nd.gy'))

# ============================================================================
# Transformer — X: tokens (batch, seq_len) < vocab_size, y: (batch, 1)
# (seq_len=4, vocab_size=6 par défaut)
# ============================================================================
X_trf = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [0, 0, 1, 1],
    [5, 5, 4, 4],
])
y_trf = np.array([[1], [0], [0], [1]])

run("transformer", lambda: IA.train_transformer(
    X=X_trf, y=y_trf, epochs=2,
    save_path='/tmp/gy_models_quick/transformer.gy'))

run("transformer3d", lambda: IA.train_transformer3d(
    X=X_trf, y=y_trf, epochs=2,
    save_path='/tmp/gy_models_quick/transformer3d.gy'))

# ============================================================================
# GAN — pas de X/y (génératif), juste peu d'epochs
# ============================================================================
run("gan_1d", lambda: IA.train_gan_1d(
    epochs=2, save_path='/tmp/gy_models_quick/gan_1d.gy'))
run("gan_nd", lambda: IA.train_gan_nd(
    epochs=2, save_path='/tmp/gy_models_quick/gan_nd.gy'))
run("gan_3d", lambda: IA.train_gan_3d(
    epochs=2, save_path='/tmp/gy_models_quick/gan_3d.gy'))

# ============================================================================
# LDM — pas de X/y, juste peu d'epochs
# ============================================================================
run("ldm_image", lambda: IA.train_ldm_image(
    epochs=2, save_path='/tmp/gy_models_quick/ldm_image.gy'))
run("ldm_audio", lambda: IA.train_ldm_audio(
    epochs=2, save_path='/tmp/gy_models_quick/ldm_audio.gy'))

# ============================================================================
# SLM emotion — sentences: liste de str, labels: liste d'indices de classe
# (6 classes : joie, tristesse, colère, peur, surprise, neutre -> 0..5)
# ============================================================================
sentences_slm = [
    "je suis tellement content aujourd'hui",
    "je suis vraiment heureux",
    "je me sens triste et seul",
    "quelle tristesse profonde",
    "ça me met en colère",
    "je suis furieux contre toi",
]
labels_slm = [0, 0, 1, 1, 2, 2]

run("slm_emotion", lambda: IA.train_slm_emotion(
    sentences=sentences_slm, labels=labels_slm, epochs=2,
    save_path='/tmp/gy_models_quick/slm_emotion.gy'))

# ============================================================================
# Vision — X: (n_samples, feature_dim), y: (n_samples,)
# feature_dim=128 (image), 26 (parole) par défaut
# ============================================================================
X_img = np.random.rand(6, 128)
y_img = np.array([0, 1, 2, 0, 1, 2])

run("image_classifier", lambda: IA.train_image_classifier(
    X=X_img, y=y_img, epochs=2,
    save_path='/tmp/gy_models_quick/image_clf.gy'))

X_speech = np.random.rand(6, 26)
y_speech = np.array([0, 1, 2, 0, 1, 2])

run("speech_classifier", lambda: IA.train_speech_classifier(
    X=X_speech, y=y_speech, epochs=2,
    save_path='/tmp/gy_models_quick/speech_clf.gy'))

# ============================================================================
# RESUME
# ============================================================================
print("\n" + "=" * 60)
print("RESUME")
print("=" * 60)

n_ok = sum(1 for status, _ in results.values() if status == "OK")
n_total = len(results)

for k, (status, err) in results.items():
    line = f"{status:5s} {k}"
    if err:
        line += f"  -> {err}"
    print(line)

print(f"\n{n_ok}/{n_total} tests réussis")
sys.exit(0 if n_ok == n_total else 1)
