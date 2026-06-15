import numpy as np
from env.constants import (
    CELL_PATH, CELL_CROP, CELL_WALL,
    STATE_UNKNOWN, CROP_STATE_PROBS, CROP_STATE_VALUES
)


def generate_field_map(n_beds: int = 4, field_height: int = 8) -> np.ndarray:
    """
    2열 재배단과 수직 주행 레인으로 구성된 온실 맵을 생성한다.

    각 재배단은 두 열 너비이며 레인이 재배단 사이를 구분한다. 로봇은 현재
    레인에 인접한 작물 열만 예찰할 수 있으므로, 바깥쪽 열을 처리하려면
    반대편 레인을 방문해야 한다.

    H = field_height + 4
    W = 3*n_beds + 3  (wall + [lane + bed(2)] * n_beds + lane + wall)

    내부 열 패턴(col 1..W-2):
      (col-1) % 3 == 0  → CELL_PATH  (주행 레인)
      (col-1) % 3 == 1  → CELL_CROP  (재배단 안쪽 열)
      (col-1) % 3 == 2  → CELL_CROP  (재배단 바깥쪽 열)

    예시 n_beds=4: W P CC P CC P CC P CC P W  (W=15)
    """
    if n_beds < 1 or field_height < 1:
        raise ValueError(
            f"n_beds and field_height must be >= 1, got n_beds={n_beds}, field_height={field_height}"
        )
    H = field_height + 4
    W = 3 * n_beds + 3
    layout = np.full((H, W), CELL_WALL, dtype=np.int32)

    layout[1, 1:-1] = CELL_PATH       # 상단 헤드랜드
    layout[-2, 1:-1] = CELL_PATH      # 하단 헤드랜드

    for row in range(2, H - 2):       # 실제 작물이 배치되는 필드 행
        for col in range(1, W - 1):
            layout[row, col] = CELL_PATH if (col - 1) % 3 == 0 else CELL_CROP

    return layout


def init_crop_states(layout: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    모든 작물 셀에 확률 분포를 따라 실제 상태를 배정한다.

    작물이 아닌 셀은 STATE_UNKNOWN(0)으로 유지한다.
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.zeros(layout.shape, dtype=np.int32)
    crop_positions = np.argwhere(layout == CELL_CROP)

    chosen = rng.choice(CROP_STATE_VALUES, size=len(crop_positions), p=CROP_STATE_PROBS)
    for (r, c), state in zip(crop_positions, chosen):
        states[r, c] = state

    return states
