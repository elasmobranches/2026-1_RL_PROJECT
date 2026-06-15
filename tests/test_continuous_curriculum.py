from env.continuous_farm_env_curriculum import CurriculumCallback


class _CurriculumEnv:
    def __init__(self):
        self.curriculum_level = 0
        self.unwrapped = self


class _VecEnv:
    def __init__(self):
        self.envs = [_CurriculumEnv()]


def test_curriculum_uses_fresh_results_for_each_level():
    vec_env = _VecEnv()
    callback = CurriculumCallback(
        vec_env,
        success_threshold=0.7,
        window=2,
        level_up_at=1.0,
        verbose=0,
    )

    callback.locals = {
        "dones": [True, True],
        "infos": [{"coverage": 1.0}, {"coverage": 1.0}],
    }
    callback._on_step()

    assert vec_env.envs[0].curriculum_level == 1
    assert callback._coverages == []

    callback.locals = {"dones": [], "infos": []}
    callback._on_step()
    assert vec_env.envs[0].curriculum_level == 1
