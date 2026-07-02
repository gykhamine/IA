"""
smoke_test.py — Test end-to-end du framework IA (entraînement + inférence).

Usage :
    Placer ce script au même niveau que le dossier IA/ (ou ajuster IA_PATH
    ci-dessous), puis :

        python3 smoke_test.py

Le script entraîne un petit modèle pour chacune des 8 familles (MLP, RNN,
CNN2D/ND, Transformer/3D, GAN 1D/ND/3D, LDM image/audio, SLM, Vision), les
sauvegarde dans /tmp/gy_models, puis recharge chaque modèle et fait une
prédiction/génération pour vérifier le cycle complet save -> load -> infer.
Un résumé OK/FAIL est affiché à la fin.
"""

import os
import sys
import traceback

# Ajuste ce chemin si le dossier IA/ n'est pas au même niveau que ce script.
IA_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IA_PATH)

os.makedirs('/tmp/gy_models', exist_ok=True)

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
# 1. ENTRAINEMENT — une passe rapide (3 epochs) pour chaque famille
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1 : ENTRAINEMENT")
print("=" * 70)

run("train_mlp", lambda: IA.train_mlp(epochs=3, save_path='/tmp/gy_models/mlp.gy'))
run("train_rnn", lambda: IA.train_rnn(epochs=3, save_path='/tmp/gy_models/rnn.gy'))
run("train_cnn2d", lambda: IA.train_cnn2d(epochs=3, save_path='/tmp/gy_models/cnn2d.gy'))
run("train_cnn_nd", lambda: IA.train_cnn_nd(epochs=3, save_path='/tmp/gy_models/cnn_nd.gy'))
run("train_transformer", lambda: IA.train_transformer(epochs=3, save_path='/tmp/gy_models/transformer.gy'))
run("train_transformer3d", lambda: IA.train_transformer3d(epochs=3, save_path='/tmp/gy_models/transformer3d.gy'))
run("train_gan_1d", lambda: IA.train_gan_1d(epochs=3, save_path='/tmp/gy_models/gan_1d.gy'))
run("train_gan_nd", lambda: IA.train_gan_nd(epochs=3, save_path='/tmp/gy_models/gan_nd.gy'))
run("train_gan_3d", lambda: IA.train_gan_3d(epochs=3, save_path='/tmp/gy_models/gan_3d.gy'))
run("train_ldm_image", lambda: IA.train_ldm_image(epochs=3, save_path='/tmp/gy_models/ldm_image.gy'))
run("train_ldm_audio", lambda: IA.train_ldm_audio(epochs=3, save_path='/tmp/gy_models/ldm_audio.gy'))
run("train_slm_next_word", lambda: IA.train_slm_next_word(epochs=3, save_path='/tmp/gy_models/slm_next_word.gy'))
run("train_slm_emotion", lambda: IA.train_slm_emotion(epochs=3, save_path='/tmp/gy_models/slm_emotion.gy'))
run("train_image_classifier", lambda: IA.train_image_classifier(epochs=3, save_path='/tmp/gy_models/image_clf.gy'))
run("train_speech_classifier", lambda: IA.train_speech_classifier(epochs=3, save_path='/tmp/gy_models/speech_clf.gy'))

# ============================================================================
# 2. INFERENCE — recharge chaque modèle sauvegardé et fait une prédiction
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2 : INFERENCE")
print("=" * 70)

run("infer_mlp", lambda: IA.predict_mlp(IA.load_mlp('/tmp/gy_models/mlp.gy'), np.random.rand(1, 4)))
run("infer_rnn", lambda: IA.predict_rnn(IA.load_rnn('/tmp/gy_models/rnn.gy'), np.random.rand(5, 2)))
run("infer_cnn2d", lambda: IA.predict_cnn2d(IA.load_cnn2d('/tmp/gy_models/cnn2d.gy'), np.random.rand(5, 5)))
run("infer_cnn_nd", lambda: IA.predict_cnn_nd(IA.load_cnn_nd('/tmp/gy_models/cnn_nd.gy'), np.random.rand(3, 3, 3, 3)))
run("infer_transformer", lambda: IA.predict_transformer(IA.load_transformer('/tmp/gy_models/transformer.gy'), [1, 2, 3, 4]))
run("infer_transformer3d", lambda: IA.predict_transformer3d(IA.load_transformer3d('/tmp/gy_models/transformer3d.gy'), [1, 2, 3, 4]))
run("infer_gan_1d", lambda: IA.generate_gan_1d(IA.load_gan_1d('/tmp/gy_models/gan_1d.gy'), num_samples=2))
run("infer_gan_nd", lambda: IA.generate_gan_nd(IA.load_gan_nd('/tmp/gy_models/gan_nd.gy'), num_samples=2))
run("infer_gan_3d", lambda: IA.generate_gan_3d(IA.load_gan_3d('/tmp/gy_models/gan_3d.gy')))
run("infer_ldm_image", lambda: IA.generate_ldm_image(IA.load_ldm_image('/tmp/gy_models/ldm_image.gy'), class_id=0))
run("infer_ldm_audio", lambda: IA.generate_ldm_audio(IA.load_ldm_audio('/tmp/gy_models/ldm_audio.gy'), class_id=0))
run("infer_slm_next_word", lambda: IA.predict_slm_next_word(IA.load_slm('/tmp/gy_models/slm_next_word.gy'), [1, 2, 3, 4]))
run("infer_image_classifier", lambda: IA.predict_image(IA.load_image_classifier('/tmp/gy_models/image_clf.gy'), np.random.rand(128)))
run("infer_speech_classifier", lambda: IA.predict_speech(IA.load_speech_classifier('/tmp/gy_models/speech_clf.gy'), np.random.rand(26)))

# ============================================================================
# 3. RESUME
# ============================================================================
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)

n_ok = sum(1 for status, _ in results.values() if status == "OK")
n_total = len(results)

for k, (status, err) in results.items():
    line = f"{status:5s} {k}"
    if err:
        line += f"  -> {err}"
    print(line)

print(f"\n{n_ok}/{n_total} tests réussis")
sys.exit(0 if n_ok == n_total else 1)
