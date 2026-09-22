from inverse_em.scientific.roles import ScientificRole


def test_roles_are_distinct():
    assert len(ScientificRole) == 8
    assert ScientificRole.VALIDATION is not ScientificRole.SEALED_TEST
    assert ScientificRole.DEVELOPMENT is not ScientificRole.TRAINING

