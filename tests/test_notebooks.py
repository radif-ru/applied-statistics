"""Проверки ноутбуков: сохранённые выводы актуальны, воспроизводимы и полны."""

import copy
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
PLACEHOLDERS = ("<ВАШ ОТВЕТ>", "# ВАШЕ РЕШЕНИЕ")
EXPECTED = {
    "statistical-tests.ipynb": [
        "z-статистика: 2.3570",
        "p-value: 0.0184",
        "95% доверительный интервал для μ: [3.00084; 3.00916] г",
        "Объединённый t-тест: t = 0.3171, p-value = 0.7541",
        "95% доверительный интервал для разности: [-3.2428; 4.4168] MPa",
        "Парный t-тест: t = -7.9649, p-value = 1.15e-05",
        "ANOVA: F = 49.9246, p-value = 1.80e-09",
        "C      D     15.5 0.0001  7.6524 23.3476   True",
        "z-тест для двух долей: z = 2.5803, p-value = 0.0099",
        "95% бутстрэп-интервал для разности: [0.0090; 0.0610]",
    ],
    "correlation-regression.ipynb": [
        "Пирсон: r = 0.9435, p-value = 5.80e-15",
        "Спирмен: r_s = -0.2108, p-value = 0.2635",
        "95% доверительный интервал для наклона: [0.7827; 1.0290]",
        "95% предсказательный интервал для нового наблюдения: [61.53; 74.53] кг",
        "R^2 = 0.8741",
        "RMSE модели = 2.8797",
        "95% перцентильный бутстрэп-интервал: [0.9021; 0.9705]",
        "Шапиро-Уилк для остатков: p-value = 0.9616",
    ],
}


def text_outputs(nb: nbformat.NotebookNode) -> list[str]:
    """Текстовый вывод каждой кодовой ячейки (stream и text/plain).

    Поток склеивается: ядро может по-разному делить его на фрагменты.
    """
    result = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        texts = []
        for out in cell.outputs:
            if out.output_type == "stream":
                texts.append(out.text)
            elif "data" in out and "image/png" not in out.data:
                texts.append(out.data.get("text/plain", ""))
        result.append("".join(texts))
    return result


def all_text(nb: nbformat.NotebookNode) -> str:
    """Весь текст ноутбука: исходники ячеек и текстовые выводы."""
    sources = [cell.source for cell in nb.cells]
    return "\n".join(sources + text_outputs(nb))


class NotebookTestMixin:
    """Общие проверки; имя ноутбука задаёт подкласс."""

    name: str
    saved: nbformat.NotebookNode
    fresh: nbformat.NotebookNode

    @classmethod
    def setUpClass(cls) -> None:
        cls.saved = nbformat.read(NOTEBOOKS / cls.name, as_version=4)
        cls.fresh = copy.deepcopy(cls.saved)
        NotebookClient(
            cls.fresh,
            timeout=600,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(NOTEBOOKS)}},
        ).execute()

    def test_saved_cells_executed_in_order(self) -> None:
        code = [c for c in self.saved.cells if c.cell_type == "code"]
        self.assertEqual(
            [c.execution_count for c in code], list(range(1, len(code) + 1))
        )
        for cell in code:
            self.assertTrue(cell.outputs or "import" in cell.source, cell.source)
            for out in cell.outputs:
                self.assertNotEqual(out.output_type, "error", cell.source)
                self.assertNotEqual(out.get("name"), "stderr", cell.source)

    def test_saved_outputs_match_fresh_run(self) -> None:
        self.assertEqual(text_outputs(self.saved), text_outputs(self.fresh))

    def test_no_placeholders_timings_or_local_paths(self) -> None:
        text = all_text(self.saved)
        for marker in (*PLACEHOLDERS, "/home/", "/Users/", "C:\\"):
            self.assertNotIn(marker, text)
        for cell in self.saved.cells:
            self.assertEqual(cell.metadata, {}, cell.source)

    def test_every_task_has_conclusion(self) -> None:
        headers = [c for c in self.saved.cells if c.source.startswith("### Задание")]
        conclusions = [c for c in self.saved.cells if c.source.startswith("**Вывод:**")]
        self.assertEqual(len(headers), len(conclusions))

    def test_key_results(self) -> None:
        text = all_text(self.saved)
        for expected in EXPECTED[self.name]:
            self.assertIn(expected, text)


class StatisticalTestsNotebook(NotebookTestMixin, unittest.TestCase):
    name = "statistical-tests.ipynb"


class CorrelationRegressionNotebook(NotebookTestMixin, unittest.TestCase):
    name = "correlation-regression.ipynb"


if __name__ == "__main__":
    unittest.main()
