from typing import Any, Literal, Type, Union, get_args
from pydantic import Field, create_model
from pydantic.fields import FieldInfo
import datetime
import uuid


class JSONSchemaConverter:
    """Converts JSON Schema definitions to Python/Pydantic types."""

    def json_schema_to_pydantic(
        self, tool_name: str, json_schema: dict[str, Any]
    ) -> Type[Any]:
        """Convert JSON Schema to Pydantic model for tool arguments.

        Args:
            tool_name: Name of the tool (used for model naming)
            json_schema: JSON Schema dict with 'properties', 'required', etc.

        Returns:
            Pydantic BaseModel class
        """
        properties = json_schema.get("properties", {})
        required_fields = json_schema.get("required", [])

        model_name = f"{tool_name.replace('-', '_').replace(' ', '_')}Schema"
        return self._create_pydantic_model(model_name, properties, required_fields)

    def _json_type_to_python(
        self, field_schema: dict[str, Any], field_name: str = "Field"
    ) -> Type[Any]:
        """Convert JSON Schema type to Python type, handling nested structures.

        Args:
            field_schema: JSON Schema field definition
            field_name: Name of the field (used for nested model naming)

        Returns:
            Python type (may be a dynamically created Pydantic model for objects/arrays)
        """
        if not field_schema:
            return Any

        # Handle $ref if needed
        if "$ref" in field_schema:
            # You might want to implement reference resolution here
            return Any

        # Handle enum constraint - create Literal type
        if "enum" in field_schema:
            return self._handle_enum(field_schema)

        # Handle different schema constructs in order of precedence
        if "allOf" in field_schema:
            return self._handle_allof(field_schema, field_name)

        if "anyOf" in field_schema or "oneOf" in field_schema:
            return self._handle_union_schemas(field_schema, field_name)

        json_type = field_schema.get("type")

        if isinstance(json_type, list):
            return self._handle_type_union(json_type)

        if json_type == "array":
            return self._handle_array_type(field_schema, field_name)

        if json_type == "object":
            return self._handle_object_type(field_schema, field_name)

        # Handle format for string types
        if json_type == "string" and "format" in field_schema:
            return self._get_formatted_type(field_schema["format"])

        return self._get_simple_type(json_type)

    def _get_formatted_type(self, format_type: str) -> Type[Any]:
        """Get Python type for JSON Schema format constraint.

        Args:
            format_type: JSON Schema format string (date, date-time, email, etc.)

        Returns:
            Appropriate Python type for the format
        """
        format_mapping: dict[str, Type[Any]] = {
            "date": datetime.date,
            "date-time": datetime.datetime,
            "time": datetime.time,
            "email": str,  # Could use EmailStr from pydantic
            "uri": str,
            "uuid": str,  # Could use UUID
            "hostname": str,
            "ipv4": str,
            "ipv6": str,
        }
        return format_mapping.get(format_type, str)

    def _handle_enum(self, field_schema: dict[str, Any]) -> Type[Any]:
        """Handle enum constraint by creating a Literal type.

        Args:
            field_schema: Schema containing enum values

        Returns:
            Literal type with enum values
        """
        enum_values = field_schema.get("enum", [])

        if not enum_values:
            return str

        # Filter out None values for the Literal type
        non_null_values = [v for v in enum_values if v is not None]

        if not non_null_values:
            return type(None)

        # Create Literal type with enum values
        # For strings, create Literal["value1", "value2", ...]
        if all(isinstance(v, str) for v in non_null_values):
            literal_type = Literal[tuple(non_null_values)]  # type: ignore[valid-type]
            # If null is in enum, make it optional
            if None in enum_values:
                return literal_type | None  # type: ignore[return-value]
            return literal_type  # type: ignore[return-value]

        # For mixed types or non-strings, fall back to the base type
        json_type = field_schema.get("type", "string")
        return self._get_simple_type(json_type)

    def _handle_allof(
        self, field_schema: dict[str, Any], field_name: str
    ) -> Type[Any]:
        """Handle allOf schema composition by merging all schemas.

        Args:
            field_schema: Schema containing allOf
            field_name: Name for the generated model

        Returns:
            Merged Pydantic model or basic type
        """
        merged_properties: dict[str, Any] = {}
        merged_required: list[str] = []
        found_type: str | None = None

        for sub_schema in field_schema["allOf"]:
            # Collect type information
            if sub_schema.get("type"):
                found_type = sub_schema.get("type")

            # Merge properties
            if sub_schema.get("properties"):
                merged_properties.update(sub_schema["properties"])

            # Merge required fields
            if sub_schema.get("required"):
                merged_required.extend(sub_schema["required"])

            # Handle nested anyOf/oneOf - merge properties from all variants
            for union_key in ("anyOf", "oneOf"):
                if union_key in sub_schema:
                    for variant in sub_schema[union_key]:
                        if variant.get("properties"):
                            # Merge variant properties (will be optional)
                            for prop_name, prop_schema in variant["properties"].items():
                                if prop_name not in merged_properties:
                                    merged_properties[prop_name] = prop_schema

        # If we found properties, create a merged object model
        if merged_properties:
            return self._create_pydantic_model(
                field_name, merged_properties, merged_required
            )

        # Fallback: return the found type or dict
        if found_type == "object":
            return dict
        elif found_type == "array":
            return list
        return dict  # Default for complex allOf

    def _handle_union_schemas(
        self, field_schema: dict[str, Any], field_name: str
    ) -> Type[Any]:
        """Handle anyOf/oneOf union schemas.

        Args:
            field_schema: Schema containing anyOf or oneOf
            field_name: Name for nested types

        Returns:
            Union type combining all options
        """
        key = "anyOf" if "anyOf" in field_schema else "oneOf"
        types: list[Type[Any]] = []

        for option in field_schema[key]:
            if "const" in option:
                # For const values, use string type
                # Could use Literal[option["const"]] for more precision
                types.append(str)
            else:
                types.append(self._json_type_to_python(option, field_name))

        return self._build_union_type(types)

    def _handle_type_union(self, json_types: list[str]) -> Type[Any]:
        """Handle union types from type arrays.

        Args:
            json_types: List of JSON Schema type strings

        Returns:
            Union of corresponding Python types
        """
        type_mapping: dict[str, Type[Any]] = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "null": type(None),
            "array": list,
            "object": dict,
        }

        types = [type_mapping.get(t, Any) for t in json_types]
        return self._build_union_type(types)

    def _handle_array_type(
        self, field_schema: dict[str, Any], field_name: str
    ) -> Type[Any]:
        """Handle array type with typed items.

        Args:
            field_schema: Schema with type="array"
            field_name: Name for item types

        Returns:
            list or list[ItemType]
        """
        items_schema = field_schema.get("items")
        if items_schema:
            item_type = self._json_type_to_python(items_schema, f"{field_name}Item")
            return list[item_type]  # type: ignore[valid-type]
        return list

    def _handle_object_type(
        self, field_schema: dict[str, Any], field_name: str
    ) -> Type[Any]:
        """Handle object type with properties.

        Args:
            field_schema: Schema with type="object"
            field_name: Name for the generated model

        Returns:
            Pydantic model or dict
        """
        properties = field_schema.get("properties")
        if properties:
            required_fields = field_schema.get("required", [])
            return self._create_pydantic_model(field_name, properties, required_fields)

        # Object without properties (e.g., additionalProperties only)
        return dict

    def _create_pydantic_model(
        self,
        field_name: str,
        properties: dict[str, Any],
        required_fields: list[str],
    ) -> Type[Any]:
        """Create a Pydantic model from properties.

        Args:
            field_name: Base name for the model
            properties: Property schemas
            required_fields: List of required property names

        Returns:
            Dynamically created Pydantic model
        """
        model_name = f"Generated_{field_name}_{uuid.uuid4().hex[:8]}"
        field_definitions: dict[str, Any] = {}

        for prop_name, prop_schema in properties.items():
            prop_type = self._json_type_to_python(prop_schema, prop_name.title())
            prop_description = self._build_field_description(prop_schema)
            is_required = prop_name in required_fields

            if is_required:
                field_definitions[prop_name] = (
                    prop_type,
                    Field(..., description=prop_description),
                )
            else:
                field_definitions[prop_name] = (
                    prop_type | None,
                    Field(default=None, description=prop_description),
                )

        return create_model(model_name, **field_definitions)  # type: ignore[return-value]

    def _build_field_description(self, prop_schema: dict[str, Any]) -> str:
        """Build a comprehensive field description including constraints.

        Args:
            prop_schema: Property schema with description and constraints

        Returns:
            Enhanced description with format, enum, and other constraints
        """
        parts: list[str] = []

        # Start with the original description
        description = prop_schema.get("description", "")
        if description:
            parts.append(description)

        # Add format constraint
        format_type = prop_schema.get("format")
        if format_type:
            parts.append(f"Format: {format_type}")

        # Add enum constraint (if not already handled by Literal type)
        enum_values = prop_schema.get("enum")
        if enum_values:
            enum_str = ", ".join(repr(v) for v in enum_values)
            parts.append(f"Allowed values: [{enum_str}]")

        # Add pattern constraint
        pattern = prop_schema.get("pattern")
        if pattern:
            parts.append(f"Pattern: {pattern}")

        # Add min/max constraints
        minimum = prop_schema.get("minimum")
        maximum = prop_schema.get("maximum")
        if minimum is not None:
            parts.append(f"Minimum: {minimum}")
        if maximum is not None:
            parts.append(f"Maximum: {maximum}")

        min_length = prop_schema.get("minLength")
        max_length = prop_schema.get("maxLength")
        if min_length is not None:
            parts.append(f"Min length: {min_length}")
        if max_length is not None:
            parts.append(f"Max length: {max_length}")

        # Add examples if available
        examples = prop_schema.get("examples")
        if examples:
            examples_str = ", ".join(repr(e) for e in examples[:3])  # Limit to 3
            parts.append(f"Examples: {examples_str}")

        return ". ".join(parts) if parts else ""

    def _get_simple_type(self, json_type: str | None) -> Type[Any]:
        """Map simple JSON Schema types to Python types.

        Args:
            json_type: JSON Schema type string

        Returns:
            Corresponding Python type
        """
        simple_type_mapping: dict[str | None, Type[Any]] = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }

        return simple_type_mapping.get(json_type, Any)

    def _build_union_type(self, types: list[Type[Any]]) -> Type[Any]:
        """Build a union type from a list of types.

        Args:
            types: List of Python types to combine

        Returns:
            Union type or single type if only one unique type
        """
        # Remove duplicates while preserving order
        unique_types = list(dict.fromkeys(types))

        if len(unique_types) == 1:
            return unique_types[0]

        # Build union using | operator
        result = unique_types[0]
        for t in unique_types[1:]:
            result = result | t
        return result  # type: ignore[no-any-return]
