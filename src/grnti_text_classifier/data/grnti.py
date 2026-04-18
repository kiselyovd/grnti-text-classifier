"""GRNTI dataset helpers: loader, label encoder, stratified split."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

FEATURES: list[str] = ["id", "label", "text"]
TEXT_COL: str = "text"
LABEL_COL: str = "label"
ENCODED_COL: str = "label_idx"

# Hardcoded map: (code // 10000) → canonical Russian section name.
# Source: standard GRNTI-2001 top-level sections.
_SECTION_NAMES: dict[int, str] = {
    2: "Философия",
    3: "История. Исторические науки",
    4: "Социология",
    5: "Демография",
    6: "Экономика. Экономические науки",
    10: "Государство и право. Юридические науки",
    11: "Политика. Политические науки",
    12: "Науковедение",
    13: "Культура. Культурология",
    14: "Народное образование. Педагогика",
    15: "Психология",
    16: "Языкознание",
    17: "Литература. Литературоведение. Устное народное творчество",
    18: "Искусство",
    19: "Массовая коммуникация. Журналистика. Средства массовой информации",
    20: "Информатика",
    21: "Религия. Атеизм",
    23: "Комплексное изучение отдельных стран и регионов",
    27: "Математика",
    28: "Кибернетика",
    29: "Физика",
    30: "Механика",
    31: "Химия",
    34: "Биология",
    36: "Геодезия. Картография",
    37: "Геофизика",
    38: "Геология",
    39: "География",
    41: "Астрономия",
    43: "Общие и комплексные проблемы естественных и точных наук",
    44: "Энергетика",
    45: "Электротехника",
    47: "Электроника. Радиотехника",
    49: "Связь",
    50: "Автоматика. Вычислительная техника",
    52: "Горное дело",
    53: "Металлургия",
    55: "Машиностроение",
    58: "Ядерная техника",
    59: "Приборостроение",
    60: "Полиграфия. Репрография. Фотокинотехника",
    61: "Химическая технология. Химическая промышленность",
    62: "Биотехнология",
    64: "Легкая промышленность",
    65: "Пищевая промышленность",
    66: "Лесная и деревообрабатывающая промышленность",
    67: "Строительство. Архитектура",
    68: "Сельское и лесное хозяйство",
    69: "Рыбное хозяйство. Аквакультура",
    70: "Водное хозяйство",
    71: "Внутренняя торговля. Туристско-экскурсионное обслуживание",
    72: "Внешняя торговля",
    73: "Транспорт",
    75: "Жилищно-коммунальное хозяйство. Домоводство. Бытовое обслуживание",
    76: "Медицина и здравоохранение",
    77: "Физическая культура и спорт",
    78: "Военное дело",
    80: "Прочие отрасли экономики",
    81: "Общие и комплексные проблемы технических и прикладных наук и отраслей народного хозяйства",
    82: "Организация и управление",
    83: "Статистика",
    84: "Стандартизация",
    85: "Патентное дело. Изобретательство. Рационализаторство",
    86: "Охрана труда",
    87: "Охрана окружающей среды. Экология человека",
    89: "Космические исследования",
    90: "Метрология",
    94: "Комплексные проблемы общественных наук",
}


def _code_to_text(code: int) -> str:
    """Return Russian section name for a raw label code, or fallback string."""
    section = code // 10000
    return _SECTION_NAMES.get(section, f"GRNTI-{code}")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a JSONL file and return a DataFrame with FEATURES columns only."""
    df = pd.read_json(path, lines=True)
    # Keep only known FEATURES columns; drop unexpected extras defensively.
    cols = [c for c in FEATURES if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------


@dataclass
class LabelEncoder:
    """Bidirectional map between raw GRNTI codes and dense 0..N-1 indices."""

    code_to_idx: dict[int, int]
    idx_to_code: dict[int, int]
    idx_to_text: dict[int, str]
    num_classes: int

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, labels: "pd.Series | list[int]") -> np.ndarray:
        """Map raw codes → dense indices."""
        return np.array([self.code_to_idx[int(c)] for c in labels], dtype=np.int64)

    def decode(self, idx: int) -> int:
        """Map dense index → raw code."""
        return self.idx_to_code[int(idx)]

    def decode_text(self, idx: int) -> str:
        """Map dense index → human-readable Russian class name."""
        return self.idx_to_text[int(idx)]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable dict."""
        return {
            "code_to_idx": {str(k): v for k, v in self.code_to_idx.items()},
            "idx_to_code": {str(k): v for k, v in self.idx_to_code.items()},
            "idx_to_text": {str(k): v for k, v in self.idx_to_text.items()},
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "LabelEncoder":
        """Reconstruct a LabelEncoder from a JSON dict."""
        return cls(
            code_to_idx={int(k): int(v) for k, v in d["code_to_idx"].items()},
            idx_to_code={int(k): int(v) for k, v in d["idx_to_code"].items()},
            idx_to_text={int(k): str(v) for k, v in d["idx_to_text"].items()},
            num_classes=int(d["num_classes"]),
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_label_encoder(df: pd.DataFrame) -> LabelEncoder:
    """Build a LabelEncoder from unique codes in *df[LABEL_COL]*.

    Codes are sorted ascending; idx = position in sorted order.
    """
    codes: list[int] = sorted(int(c) for c in df[LABEL_COL].unique())
    code_to_idx: dict[int, int] = {c: i for i, c in enumerate(codes)}
    idx_to_code: dict[int, int] = {i: c for i, c in enumerate(codes)}
    idx_to_text: dict[int, str] = {i: _code_to_text(c) for i, c in enumerate(codes)}
    return LabelEncoder(
        code_to_idx=code_to_idx,
        idx_to_code=idx_to_code,
        idx_to_text=idx_to_text,
        num_classes=len(codes),
    )


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


def split_stratified_train_val(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train and val subsets with stratification on LABEL_COL."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df[LABEL_COL],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
