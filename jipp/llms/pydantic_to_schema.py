from typing import Any, Dict, Union
import pydantic


def pydantic_model_to_openai_schema(
    model: Union[type[pydantic.BaseModel], pydantic.TypeAdapter[Any]]
) -> Dict[str, Any]:
    """Convert a Pydantic model or TypeAdapter to OpenAI's strict JSON schema format."""

    def to_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        return _ensure_strict_json_schema(schema, path=(), root=schema)

    def _ensure_strict_json_schema(
        json_schema: Dict[str, Any],
        *,
        path: tuple[str, ...],
        root: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recursively ensure the JSON schema is strict and follows OpenAI's format."""
        if not isinstance(json_schema, dict):
            raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

        # Handle $defs
        defs = json_schema.get("$defs")
        if isinstance(defs, dict):
            for def_name, def_schema in defs.items():
                _ensure_strict_json_schema(
                    def_schema, path=(*path, "$defs", def_name), root=root
                )

        typ = json_schema.get("type")
        if typ == "object" and "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False

        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            json_schema["required"] = list(properties.keys())
            json_schema["properties"] = {
                key: _ensure_strict_json_schema(
                    prop_schema, path=(*path, "properties", key), root=root
                )
                for key, prop_schema in properties.items()
            }

        items = json_schema.get("items")
        if isinstance(items, dict):
            json_schema["items"] = _ensure_strict_json_schema(
                items, path=(*path, "items"), root=root
            )

        any_of = json_schema.get("anyOf")
        if isinstance(any_of, list):
            json_schema["anyOf"] = [
                _ensure_strict_json_schema(
                    variant, path=(*path, "anyOf", str(i)), root=root
                )
                for i, variant in enumerate(any_of)
            ]

        all_of = json_schema.get("allOf")
        if isinstance(all_of, list):
            if len(all_of) == 1:
                json_schema.update(
                    _ensure_strict_json_schema(
                        all_of[0], path=(*path, "allOf", "0"), root=root
                    )
                )
                json_schema.pop("allOf")
            else:
                json_schema["allOf"] = [
                    _ensure_strict_json_schema(
                        entry, path=(*path, "allOf", str(i)), root=root
                    )
                    for i, entry in enumerate(all_of)
                ]

        if json_schema.get("default") is None:
            json_schema.pop("default", None)

        ref = json_schema.get("$ref")
        if ref and len(json_schema) > 1:
            resolved = _resolve_ref(root=root, ref=ref)
            if not isinstance(resolved, dict):
                raise ValueError(
                    f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
                )
            json_schema.update({**resolved, **json_schema})
            json_schema.pop("$ref")

        return json_schema

    def _resolve_ref(*, root: Dict[str, Any], ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

        path = ref[2:].split("/")
        resolved = root
        for key in path:
            resolved = resolved[key]
            if not isinstance(resolved, dict):
                raise ValueError(
                    f"encountered non-dictionary entry while resolving {ref} - {resolved}"
                )

        return resolved

    if isinstance(model, type) and issubclass(model, pydantic.BaseModel):
        schema = model.model_json_schema()
        name = model.__name__
    elif isinstance(model, pydantic.TypeAdapter):
        schema = model.json_schema()
        name = getattr(model.type_, "__name__", "")
    else:
        raise TypeError(f"Unsupported type: {type(model)}")

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": to_strict_json_schema(schema),
            "name": name,
            "strict": True,
        },
    }


def pydantic_model_to_groq_schema(
    model: Union[type[pydantic.BaseModel], pydantic.TypeAdapter[Any]]
) -> Dict[str, Any]:
    """Convert a Pydantic model or TypeAdapter to OpenAI's strict JSON schema format."""

    def to_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        return _ensure_strict_json_schema(schema, path=(), root=schema)

    def _ensure_strict_json_schema(
        json_schema: Dict[str, Any],
        *,
        path: tuple[str, ...],
        root: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recursively ensure the JSON schema is strict and follows OpenAI's format."""
        if not isinstance(json_schema, dict):
            raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

        # Handle $defs
        defs = json_schema.get("$defs")
        if isinstance(defs, dict):
            for def_name, def_schema in defs.items():
                _ensure_strict_json_schema(
                    def_schema, path=(*path, "$defs", def_name), root=root
                )

        typ = json_schema.get("type")
        if typ == "object" and "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False

        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            json_schema["required"] = list(properties.keys())
            json_schema["properties"] = {
                key: _ensure_strict_json_schema(
                    prop_schema, path=(*path, "properties", key), root=root
                )
                for key, prop_schema in properties.items()
            }

        items = json_schema.get("items")
        if isinstance(items, dict):
            json_schema["items"] = _ensure_strict_json_schema(
                items, path=(*path, "items"), root=root
            )

        any_of = json_schema.get("anyOf")
        if isinstance(any_of, list):
            json_schema["anyOf"] = [
                _ensure_strict_json_schema(
                    variant, path=(*path, "anyOf", str(i)), root=root
                )
                for i, variant in enumerate(any_of)
            ]

        all_of = json_schema.get("allOf")
        if isinstance(all_of, list):
            if len(all_of) == 1:
                json_schema.update(
                    _ensure_strict_json_schema(
                        all_of[0], path=(*path, "allOf", "0"), root=root
                    )
                )
                json_schema.pop("allOf")
            else:
                json_schema["allOf"] = [
                    _ensure_strict_json_schema(
                        entry, path=(*path, "allOf", str(i)), root=root
                    )
                    for i, entry in enumerate(all_of)
                ]

        if json_schema.get("default") is None:
            json_schema.pop("default", None)

        ref = json_schema.get("$ref")
        if ref and len(json_schema) > 1:
            resolved = _resolve_ref(root=root, ref=ref)
            if not isinstance(resolved, dict):
                raise ValueError(
                    f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
                )
            json_schema.update({**resolved, **json_schema})
            json_schema.pop("$ref")

        return json_schema

    def _resolve_ref(*, root: Dict[str, Any], ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

        path = ref[2:].split("/")
        resolved = root
        for key in path:
            resolved = resolved[key]
            if not isinstance(resolved, dict):
                raise ValueError(
                    f"encountered non-dictionary entry while resolving {ref} - {resolved}"
                )

        return resolved

    if isinstance(model, type) and issubclass(model, pydantic.BaseModel):
        schema = model.model_json_schema()
        name = model.__name__
    elif isinstance(model, pydantic.TypeAdapter):
        schema = model.json_schema()
        name = getattr(model.type_, "__name__", "")
    else:
        raise TypeError(f"Unsupported type: {type(model)}")

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": to_strict_json_schema(schema),
            "name": name,
            "strict": True,
        },
    }
