from __future__ import annotations


class StaticBuildRecipe:
    def __init__(self, value):
        self.value = value

    def build(self, *args, **kwargs):
        del args, kwargs
        return self.value


def required_build_recipe(value):
    if callable(getattr(value, "build", None)):
        return value
    return StaticBuildRecipe(value)


def optional_build_recipe(value):
    if value is None or callable(getattr(value, "build", None)):
        return value
    return StaticBuildRecipe(value)


__all__ = [
    "StaticBuildRecipe",
    "optional_build_recipe",
    "required_build_recipe",
]
