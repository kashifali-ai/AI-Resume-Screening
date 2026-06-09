"""Semantic matching recognizes equivalent skills and rejects unrelated ones."""


def _sim(embedder, a, b):
    return float(embedder.similarity_matrix([a], [b])[0][0])


def test_equivalent_skills_exceed_threshold(embedder, settings):
    t = settings.sim_threshold
    assert _sim(embedder, "Spring Framework", "Spring Boot") >= t
    assert _sim(embedder, "REST Services", "REST APIs") >= t
    assert _sim(embedder, "ReactJS", "React") >= t
    assert _sim(embedder, "PostgreSQL", "Postgres") >= t


def test_related_pairs_beat_unrelated(embedder):
    # JPA ~ Hibernate should be more similar than JPA ~ an unrelated concept.
    assert _sim(embedder, "JPA", "Hibernate") > _sim(embedder, "JPA", "Photography")


def test_unrelated_skills_below_threshold(embedder, settings):
    assert _sim(embedder, "React", "PostgreSQL") < settings.sim_threshold


def _reqs(skills, kind="required", weight=1.0):
    return [{"skill": s, "kind": kind, "weight": weight} for s in skills]


def test_match_skills_maps_equivalents(embedder, settings):
    from semantic import match_skills

    candidate = ["Spring Framework", "ReactJS"]
    matches = {m["required"]: m for m in match_skills(
        _reqs(["Spring Boot", "React"]), candidate, embedder, settings.sim_threshold
    )}
    assert matches["Spring Boot"]["matched"] is True
    assert matches["React"]["matched"] is True


def test_match_skills_preserves_weight_and_kind(embedder, settings):
    from semantic import match_skills

    reqs = _reqs(["Spring Boot"], kind="preferred", weight=0.5)
    out = match_skills(reqs, ["Spring Framework"], embedder, settings.sim_threshold)
    assert out[0]["kind"] == "preferred"
    assert out[0]["weight"] == 0.5


def test_no_candidate_skills_all_unmatched(embedder, settings):
    from semantic import match_skills

    matches = match_skills(_reqs(["Python"]), [], embedder, settings.sim_threshold)
    assert matches[0]["matched"] is False
