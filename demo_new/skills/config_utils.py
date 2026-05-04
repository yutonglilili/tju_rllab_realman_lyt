import json
import os
import re


class ConfigNamespace:
    def __init__(self, values):
        self._values = dict(values)
        for key, value in self._values.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self._values)

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __getitem__(self, key):
        return self._values[key]


def resolve_config_path(
    base_path,
    filename="config.yaml",
    fallback_filenames=("config.yml", "config.json"),
):
    base_path = os.path.abspath(base_path)
    if os.path.isfile(base_path):
        base_path = os.path.dirname(base_path)

    candidate = os.path.join(base_path, filename)
    if os.path.exists(candidate):
        return candidate

    for fallback_filename in fallback_filenames:
        fallback_path = os.path.join(base_path, fallback_filename)
        if os.path.exists(fallback_path):
            return fallback_path

    return candidate


def _strip_inline_comment(line):
    in_single_quote = False
    in_double_quote = False
    escape = False

    for i, char in enumerate(line):
        if char == "\\" and in_double_quote and not escape:
            escape = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote and not escape:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()

        escape = False

    return line.rstrip()


def _preprocess_yaml_lines(text, *, source):
    lines = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue

        if raw_line.lstrip().startswith("#"):
            continue

        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]:
            raise ValueError(f"Tabs are not supported in YAML indentation: {source}:{line_no}")

        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        lines.append((indent, content, line_no))

    return lines


_INT_RE = re.compile(r"[-+]?[0-9]+$")
_FLOAT_RE = re.compile(
    r"""
    [-+]?
    (?:
        (?:[0-9]+\.[0-9]*)
        |
        (?:\.[0-9]+)
        |
        (?:[0-9]+(?:[eE][-+]?[0-9]+))
        |
        (?:[0-9]+\.[0-9]*[eE][-+]?[0-9]+)
        |
        (?:\.[0-9]+[eE][-+]?[0-9]+)
    )
    $
    """,
    re.VERBOSE,
)


def _parse_yaml_scalar(value):
    lowered = value.lower()

    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "~"}:
        return None

    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]

    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")

    if _INT_RE.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            pass

    if _FLOAT_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def _parse_yaml_mapping(lines, index, indent, *, source):
    result = {}

    while index < len(lines):
        line_indent, content, line_no = lines[index]

        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation in {source}:{line_no}")
        if content.startswith("- "):
            raise ValueError(f"YAML lists are not supported in {source}:{line_no}")
        if ":" not in content:
            raise ValueError(f"Expected 'key: value' in {source}:{line_no}")

        key, value_text = content.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()

        if not key:
            raise ValueError(f"Empty YAML key in {source}:{line_no}")

        index += 1

        if value_text:
            result[key] = _parse_yaml_scalar(value_text)
            continue

        if index < len(lines) and lines[index][0] > line_indent:
            child_indent = lines[index][0]
            child_value, index = _parse_yaml_mapping(
                lines,
                index,
                child_indent,
                source=source,
            )
            result[key] = child_value
        else:
            result[key] = None

    return result, index


def parse_yaml_config(text, *, source="<yaml>"):
    lines = _preprocess_yaml_lines(text, source=source)
    if not lines:
        return {}

    first_indent = lines[0][0]
    if first_indent != 0:
        raise ValueError(f"Root YAML indentation must start at 0: {source}:{lines[0][2]}")

    parsed, index = _parse_yaml_mapping(lines, 0, 0, source=source)
    if index != len(lines):
        raise ValueError(f"Could not fully parse YAML config: {source}")

    return parsed


def load_config_file(path, *, required=True):
    if not path:
        return {}

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}

    extension = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if extension == ".json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config file: {path}") from exc

    if extension in {".yaml", ".yml"}:
        return parse_yaml_config(content, source=path)

    raise ValueError(f"Unsupported config file extension: {path}")


def load_yaml_config(path, *, required=True):
    data = load_config_file(path, required=required)
    if data and not isinstance(data, dict):
        raise TypeError(f"YAML config must be a JSON/YAML object: {path}")
    return data


def validate_config_keys(config_data, allowed_keys=None, *, source="config"):
    if allowed_keys is None:
        return

    unknown_keys = sorted(key for key in config_data if key not in allowed_keys)
    if unknown_keys:
        raise KeyError(f"Unknown config keys in {source}: {unknown_keys}")


def validate_required_keys(config_data, required_keys=None, *, source="config"):
    if required_keys is None:
        return

    missing_keys = sorted(key for key in required_keys if key not in config_data)
    if missing_keys:
        raise KeyError(f"Missing config keys in {source}: {missing_keys}")


def extract_config_section(config_data, *, section_keys=(), allowed_keys=None, source="config"):
    if not isinstance(config_data, dict):
        raise TypeError(
            f"Config payload must be a dict, got {type(config_data).__name__} from {source}"
        )

    for section_key in section_keys or ():
        if section_key in config_data:
            section = config_data[section_key]
            if not isinstance(section, dict):
                raise TypeError(
                    f"The '{section_key}' section in {source} must be a JSON/YAML object."
                )
            validate_config_keys(section, allowed_keys, source=source)
            return section

    if allowed_keys is None:
        return dict(config_data)

    direct_config = {key: value for key, value in config_data.items() if key in allowed_keys}
    validate_config_keys(direct_config, allowed_keys, source=source)
    return direct_config


def _config_source_to_dict(config_source, *, section_keys=(), allowed_keys=None, source="config"):
    if config_source is None:
        return {}

    if hasattr(config_source, "to_dict") and callable(config_source.to_dict):
        config_dict = config_source.to_dict()
        if not isinstance(config_dict, dict):
            raise TypeError(f"to_dict() must return a dict for {source}")
        validate_config_keys(config_dict, allowed_keys, source=source)
        return config_dict

    if isinstance(config_source, str):
        config_data = load_config_file(config_source, required=False)
        if not config_data:
            return {}
        return extract_config_section(
            config_data,
            section_keys=section_keys,
            allowed_keys=allowed_keys,
            source=config_source,
        )

    return extract_config_section(
        config_source,
        section_keys=section_keys,
        allowed_keys=allowed_keys,
        source=source,
    )


def load_config_with_defaults(
    *,
    default_config_path,
    override_config_path=None,
    override_config=None,
    section_keys=(),
    allowed_keys=None,
    required_keys=None,
    config_cls=ConfigNamespace,
):
    default_config = load_config_file(default_config_path, required=True)
    if not isinstance(default_config, dict):
        raise TypeError(f"Default config must be a JSON/YAML object: {default_config_path}")

    validate_config_keys(default_config, allowed_keys, source=default_config_path)

    merged_config = dict(default_config)

    if override_config_path:
        merged_config.update(
            _config_source_to_dict(
                override_config_path,
                section_keys=section_keys,
                allowed_keys=allowed_keys,
                source=override_config_path,
            )
        )

    if override_config is not None:
        merged_config.update(
            _config_source_to_dict(
                override_config,
                section_keys=section_keys,
                allowed_keys=allowed_keys,
                source="override_config",
            )
        )

    validate_required_keys(merged_config, required_keys, source=default_config_path)

    return config_cls(merged_config)
