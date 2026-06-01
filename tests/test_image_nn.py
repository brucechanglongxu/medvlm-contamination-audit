import numpy as np
import pytest

PIL = pytest.importorskip("PIL.Image")
Image = PIL


def _make_image(tmp_path, name, color, size=64, noise_seed=None):
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        arr = np.clip(arr.astype(int) + rng.integers(-5, 6, size=arr.shape), 0, 255).astype(
            np.uint8
        )
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return path


def test_phash_identical_images_have_zero_hamming(tmp_path):
    from medvlm_contam.embedders import PHashEmbedder, pairwise_distance

    a = _make_image(tmp_path, "a.png", color=(40, 80, 200))
    b = _make_image(tmp_path, "b.png", color=(40, 80, 200))
    emb = PHashEmbedder()
    feats = emb.embed_many([a, b])
    d = pairwise_distance(feats[:1], feats[1:], metric="hamming")
    assert d[0, 0] == 0


def test_phash_distinct_images_have_high_hamming(tmp_path):
    from medvlm_contam.embedders import PHashEmbedder, pairwise_distance

    # Two visibly different images: a horizontal gradient vs random noise.
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    a, b = tmp_path / "g1.png", tmp_path / "g2.png"
    Image.fromarray(np.stack([grad] * 3, axis=-1)).save(a)
    Image.fromarray(np.stack([noise] * 3, axis=-1)).save(b)
    feats = PHashEmbedder().embed_many([a, b])
    d = pairwise_distance(feats[:1], feats[1:], metric="hamming")
    assert d[0, 0] > 5


def test_detect_image_duplicates_flags_planted_copy(tmp_path):
    from medvlm_contam.detectors.image_nn import detect_image_duplicates
    from medvlm_contam.embedders import PHashEmbedder

    # Corpus: 8 distinct gradient images.
    corpus_paths = []
    for i in range(8):
        arr = np.tile(np.linspace(i * 10, 255, 64, dtype=np.uint8), (64, 1))
        p = tmp_path / f"corpus_{i}.png"
        Image.fromarray(np.stack([arr] * 3, axis=-1)).save(p)
        corpus_paths.append(p)

    # Benchmark: 2 examples, the FIRST is a near-duplicate of corpus_3.
    dup_arr = np.tile(np.linspace(30, 255, 64, dtype=np.uint8), (64, 1))  # ~ corpus_3
    rng = np.random.default_rng(0)
    novel_arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    b1 = tmp_path / "bench_dup.png"
    b2 = tmp_path / "bench_novel.png"
    Image.fromarray(np.stack([dup_arr] * 3, axis=-1)).save(b1)
    Image.fromarray(np.stack([novel_arr] * 3, axis=-1)).save(b2)

    # Null: 12 random images (post-cutoff proxy) — not in the corpus.
    null_paths = []
    for j in range(12):
        rng = np.random.default_rng(100 + j)
        arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        p = tmp_path / f"null_{j}.png"
        Image.fromarray(np.stack([arr] * 3, axis=-1)).save(p)
        null_paths.append(p)

    flags = detect_image_duplicates(
        PHashEmbedder(),
        benchmark_image_paths=[b1, b2],
        benchmark_ids=["bench::dup", "bench::novel"],
        corpus_image_paths=corpus_paths,
        null_image_paths=null_paths,
        alpha=0.05,
    )

    assert flags.flagged_mask[0]  # near-duplicate of corpus_3 is flagged
    assert not flags.flagged_mask[1]  # novel random image is not
    assert flags.embedder_name == "phash64"
    assert flags.metric == "hamming"
    assert flags.threshold >= 0


def test_calibrate_threshold_matches_quantile():
    from medvlm_contam.detectors.image_nn import calibrate_threshold

    rng = np.random.default_rng(0)
    null = rng.uniform(0, 1, size=10_000)
    thresh, q = calibrate_threshold(null, alpha=0.05)
    # Should land near the 5%-quantile of Uniform(0,1) = 0.05.
    assert 0.03 < thresh < 0.07
    assert "q05" in q
