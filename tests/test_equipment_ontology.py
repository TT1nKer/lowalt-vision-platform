import unittest
from pathlib import Path

from equipment_data.ontology import load_equipment_ontology


class EquipmentOntologyTests(unittest.TestCase):
    def setUp(self):
        self.ontology = load_equipment_ontology(Path("equipment_data/ontology.yaml"))

    def test_required_concrete_leaf_classes_are_stable(self):
        self.assertEqual(
            self.ontology.leaf_ids,
            {
                "excavator", "wheel_loader", "backhoe_loader", "bulldozer",
                "motor_grader", "road_roller", "asphalt_paver", "dump_truck",
                "mobile_crane", "tower_crane",
            },
        )

    def test_vague_legacy_categories_abstain(self):
        for label in ("construction machinery", "engineering vehicle", "crane", "loader"):
            with self.subTest(label=label):
                self.assertIsNone(self.ontology.resolve(label))

    def test_verified_alias_can_resolve_but_legacy_prompt_cannot_auto_label(self):
        self.assertEqual(self.ontology.resolve("paver"), "asphalt_paver")
        self.assertIsNone(self.ontology.resolve("paver", source_kind="legacy_prompt"))


if __name__ == "__main__":
    unittest.main()
