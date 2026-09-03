"""Shape-equivalent candidates separated in absolute space."""

from .base import NegativeGenerator, endpoints, shape_distance


class TranslatedShapeNegativeGenerator(NegativeGenerator):
    name = "translated_shape"
    version = "1.0"

    def __init__(
        self, *, min_translation_m: float = 100.0, max_shape_distance: float = 0.10, **defaults
    ) -> None:
        super().__init__(
            min_translation_m=min_translation_m, max_shape_distance=max_shape_distance, **defaults
        )

    def _qualify(self, query, candidate, config):
        qo, _ = endpoints(query)
        co, _ = endpoints(candidate)
        displacement = float(((qo - co) ** 2).sum() ** 0.5)
        shape = shape_distance(query, candidate)
        achieved = {
            "translation_distance_m": displacement,
            "shape_distance": shape,
            "shape_distance_definition": "translation_scale_normalized_resampled_l2_v1",
        }
        if displacement < float(config["min_translation_m"]):
            return False, achieved, "translation_distance_too_small"
        if shape > float(config["max_shape_distance"]):
            return False, achieved, "shape_distance_too_large"
        return True, achieved, None


TranslatedShapeCopyNegativeGenerator = TranslatedShapeNegativeGenerator
