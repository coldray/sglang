"""Unit tests for MiMoDetector - no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.mimo_detector import (
    MiMoDetector,
    _convert_param_value,
    _get_param_type,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "exact": {"type": "boolean"},
                        "options": {"type": "object"},
                        "tags": {"type": "array"},
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                description="Search",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
    ]


def _mimo_call(name, params):
    body = "".join(
        f"<parameter={key}>{value}</parameter>" for key, value in params.items()
    )
    return f"<tool_call><function={name}>{body}</function></tool_call>"


class TestMiMoHelpers(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_get_param_type(self):
        self.assertEqual(_get_param_type("get_weather", "days", self.tools), "integer")
        self.assertEqual(
            _get_param_type("get_weather", "missing", self.tools), "string"
        )
        self.assertEqual(_get_param_type("missing", "days", self.tools), "string")

    def test_convert_param_value(self):
        cases = [
            ("Seattle &amp; nearby", "city", "Seattle & nearby"),
            ("3", "days", 3),
            ("not-an-int", "days", "not-an-int"),
            ("0.5", "ratio", 0.5),
            ("2.0", "ratio", 2),
            ("not-a-number", "ratio", "not-a-number"),
            ("true", "exact", True),
            ("false", "exact", False),
            ("invalid", "exact", False),
            ('{"unit": "c"}', "options", {"unit": "c"}),
            ("{'unit': 'c'}", "options", {"unit": "c"}),
            ('["open", "late"]', "tags", ["open", "late"]),
            ("not-json", "options", "not-json"),
            ("null", "city", None),
        ]
        for value, parameter, expected in cases:
            with self.subTest(value=value, parameter=parameter):
                self.assertEqual(
                    _convert_param_value(value, parameter, "get_weather", self.tools),
                    expected,
                )


class TestMiMoDetector(CustomTestCase):
    def setUp(self):
        self.detector = MiMoDetector()
        self.tools = _make_tools()

    def test_has_tool_call(self):
        self.assertTrue(
            self.detector.has_tool_call(_mimo_call("get_weather", {"city": "Seattle"}))
        )
        self.assertFalse(self.detector.has_tool_call("No tools are needed."))

    def test_detect_and_parse_plain_text(self):
        text = "No tools are needed."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_schema_aware_parameters(self):
        text = "Checking. " + _mimo_call(
            "get_weather",
            {
                "city": "Seattle",
                "days": "3",
                "ratio": "0.5",
                "exact": "true",
                "options": '{"unit": "c"}',
                "tags": '["open"]',
            },
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking. ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "city": "Seattle",
                "days": 3,
                "ratio": 0.5,
                "exact": True,
                "options": {"unit": "c"},
                "tags": ["open"],
            },
        )

    def test_detect_and_parse_multiple_calls(self):
        text = _mimo_call("get_weather", {"city": "Seattle"}) + _mimo_call(
            "search", {"query": "pizza"}
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "search"]
        )
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [{"city": "Seattle"}, {"query": "pizza"}],
        )

    def test_detect_and_parse_unknown_tool_as_normal_text(self):
        text = _mimo_call("missing", {"value": "1"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_parse_tool_call_malformed(self):
        self.assertIsNone(
            self.detector._parse_tool_call(
                "<function=get_weather>incomplete", self.tools
            )
        )

    def test_streaming_plain_text(self):
        result = self.detector.parse_streaming_increment(
            "A normal response.", self.tools
        )
        self.assertEqual(result.normal_text, "A normal response.")
        self.assertEqual(result.calls, [])

    def test_streaming_incomplete_call_waits(self):
        first = self.detector.parse_streaming_increment(
            "Checking. <tool_call><function=get_weather>", self.tools
        )
        second = self.detector.parse_streaming_increment(
            "<parameter=city>Seattle</parameter></function></tool_call>", self.tools
        )

        self.assertEqual(first.normal_text, "Checking. ")
        self.assertEqual(first.calls, [])
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.calls[0].name, "get_weather")
        self.assertEqual(json.loads(second.calls[0].parameters), {"city": "Seattle"})

    def test_streaming_multiple_complete_calls(self):
        text = _mimo_call("get_weather", {"city": "Seattle"}) + _mimo_call(
            "search", {"query": "pizza"}
        )
        first = self.detector.parse_streaming_increment(text, self.tools)
        second = self.detector.parse_streaming_increment("", self.tools)

        calls = first.calls + second.calls
        self.assertEqual([call.name for call in calls], ["get_weather", "search"])
        self.assertEqual([call.tool_index for call in calls], [0, 1])

    def test_streaming_unknown_tool_emits_no_call(self):
        text = _mimo_call("missing", {"value": "1"})
        result = self.detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_parser_registry(self):
        parser = FunctionCallParser(self.tools, "mimo")
        self.assertIsInstance(parser.detector, MiMoDetector)

    def test_structural_tag_is_not_supported(self):
        self.assertFalse(self.detector.supports_structural_tag())
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    unittest.main()
