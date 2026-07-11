"""Unit tests for MinimaxM2Detector - no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
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
                        "exact": {"type": "boolean"},
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
                    "properties": {
                        "query": {"type": "string"},
                        "ratio": {"type": "number"},
                        "filters": {"type": "array"},
                        "options": {"type": "object"},
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


def _minimax_call(name, params):
    body = "".join(
        f'<parameter name="{key}">{value}</parameter>' for key, value in params.items()
    )
    return f'<invoke name="{name}">{body}</invoke>'


def _minimax_block(*calls, prefix="", suffix=""):
    return (
        prefix
        + "<minimax:tool_call>"
        + "".join(calls)
        + "</minimax:tool_call>"
        + suffix
    )


class TestMinimaxM2Detector(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()
        self.tools = _make_tools()

    def test_has_tool_call(self):
        self.assertTrue(
            self.detector.has_tool_call(
                _minimax_block(_minimax_call("get_weather", {"city": "Seattle"}))
            )
        )
        self.assertFalse(self.detector.has_tool_call("No tools are needed."))

    def test_extract_types_from_schema(self):
        self.assertCountEqual(
            self.detector._extract_types_from_schema(
                {"anyOf": [{"type": "integer"}, {"type": "null"}]}
            ),
            ["integer", "null"],
        )
        self.assertCountEqual(
            self.detector._extract_types_from_schema(
                {"enum": ["fast", 2, True, None, ["tag"], {"key": "value"}]}
            ),
            ["string", "integer", "boolean", "null", "array", "object"],
        )
        self.assertEqual(self.detector._extract_types_from_schema(None), ["string"])
        self.assertEqual(
            self.detector._extract_types_from_schema("not-a-schema"), ["string"]
        )

    def test_convert_param_value_with_types(self):
        cases = [
            ("7", ["integer"], 7),
            ("2.5", ["number"], 2.5),
            ("2.0", ["number"], 2),
            ("yes", ["boolean"], True),
            ("off", ["boolean"], False),
            ('{"limit": 3}', ["object"], {"limit": 3}),
            ('["a", "b"]', ["array"], ["a", "b"]),
            ("plain text", ["string"], "plain text"),
            ("null", ["string"], None),
            ("not-json", ["object"], "not-json"),
        ]
        for value, param_types, expected in cases:
            with self.subTest(value=value, param_types=param_types):
                self.assertEqual(
                    self.detector._convert_param_value_with_types(value, param_types),
                    expected,
                )

    def test_get_param_types_from_config(self):
        config = {
            "count": {"type": ["integer", "null"]},
            "mode": {"oneOf": [{"type": "string"}, {"type": "boolean"}]},
            "invalid": "string",
        }
        self.assertCountEqual(
            self.detector._get_param_types_from_config("count", config),
            ["integer", "null"],
        )
        self.assertCountEqual(
            self.detector._get_param_types_from_config("mode", config),
            ["string", "boolean"],
        )
        self.assertEqual(
            self.detector._get_param_types_from_config("missing", config), ["string"]
        )
        self.assertEqual(
            self.detector._get_param_types_from_config("invalid", config), ["string"]
        )

    def test_detect_and_parse_plain_text(self):
        text = "No tools are needed."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_incomplete_block_as_normal_text(self):
        text = '<minimax:tool_call><invoke name="get_weather">'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_schema_aware_parameters(self):
        text = _minimax_block(
            _minimax_call(
                "get_weather", {"city": "Seattle", "days": "3", "exact": "true"}
            ),
            prefix="Checking now. ",
            suffix=" Done.",
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking now.  Done.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Seattle", "days": 3, "exact": True},
        )

    def test_detect_and_parse_multiple_calls(self):
        text = _minimax_block(
            _minimax_call("get_weather", {"city": "Seattle"}),
            _minimax_call(
                "search",
                {
                    "query": "pizza",
                    "ratio": "0.5",
                    "filters": '["open"]',
                    "options": '{"sort": "rating"}',
                },
            ),
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "search"]
        )
        self.assertEqual(
            json.loads(result.calls[1].parameters),
            {
                "query": "pizza",
                "ratio": 0.5,
                "filters": ["open"],
                "options": {"sort": "rating"},
            },
        )

    def test_detect_and_parse_unknown_tool_dropped(self):
        text = _minimax_block(_minimax_call("missing", {"value": "1"}))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_streaming_plain_text(self):
        result = self.detector.parse_streaming_increment(
            "A normal response.", self.tools
        )
        self.assertEqual(result.normal_text, "A normal response.")
        self.assertEqual(result.calls, [])

    def test_streaming_call_across_chunks(self):
        chunks = [
            'Checking. <minimax:tool_call><invoke name="get_weather">',
            '<parameter name="city">Seattle</parameter>',
            '<parameter name="days">3</parameter></invoke>',
        ]
        results = [
            self.detector.parse_streaming_increment(chunk, self.tools)
            for chunk in chunks
        ]
        calls = [call for result in results for call in result.calls]

        self.assertEqual(results[0].normal_text, "Checking. ")
        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        streamed_parameters = "".join(
            call.parameters for call in calls if call.parameters
        )
        self.assertEqual(
            json.loads(streamed_parameters), {"city": "Seattle", "days": 3}
        )

    def test_streaming_unknown_tool_emits_no_call(self):
        result = self.detector.parse_streaming_increment(
            '<minimax:tool_call><invoke name="missing">', self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertFalse(self.detector._in_tool_call)

    def test_parser_registry(self):
        parser = FunctionCallParser(self.tools, "minimax-m2")
        self.assertIsInstance(parser.detector, MinimaxM2Detector)

    def test_structural_tag_is_not_supported(self):
        self.assertFalse(self.detector.supports_structural_tag())
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    unittest.main()
