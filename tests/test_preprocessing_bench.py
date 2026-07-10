import unittest

from preprocessing.bench import _b64_sizes


class PreprocessingBenchB64Tests(unittest.TestCase):
    def test_b64_sizes_reports_per_grid_and_totals(self) -> None:
        # "YQ==" decodes to 1 byte; base64 string is 4 chars
        sizes = _b64_sizes(["YQ==", "YWI="])

        self.assertEqual(len(sizes["per_grid"]), 2)
        self.assertEqual(sizes["per_grid"][0]["b64_bytes"], 4)
        self.assertEqual(sizes["per_grid"][0]["decoded_bytes"], 1)
        self.assertEqual(sizes["per_grid"][1]["b64_bytes"], 4)
        self.assertEqual(sizes["per_grid"][1]["decoded_bytes"], 2)
        self.assertEqual(sizes["total_b64_bytes"], 8)
        self.assertEqual(sizes["total_decoded_bytes"], 3)
        self.assertEqual(sizes["avg_b64_bytes"], 4)

    def test_b64_sizes_empty_list(self) -> None:
        sizes = _b64_sizes([])
        self.assertEqual(sizes["per_grid"], [])
        self.assertEqual(sizes["total_b64_bytes"], 0)
        self.assertEqual(sizes["avg_b64_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
