"""Unit tests for Step3Detector - no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.step3_detector import (
    Step3Detector,
    get_argument_type,
    parse_arguments,
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
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
    ]


def _step3_call(name, params, call_type="function"):
    body = "".join(
        f'<steptml:parameter name="{key}">{value}</steptml:parameter>'
        for key, value in params.items()
    )
    return (
        "<｜tool_call_begin｜>"
        f"{call_type}"
        "<｜tool_sep｜>"
        f'<steptml:invoke name="{name}">'
        f"{body}"
        "</steptml:invoke>"
        "<｜tool_call_end｜>"
    )


def _step3_block(*calls, prefix="", suffix=""):
    return (
        prefix
        + "<｜tool_calls_begin｜>"
        + "".join(calls)
        + "<｜tool_calls_end｜>"
        + suffix
    )


class TestStep3Helpers(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_get_argument_type(self):
        self.assertEqual(get_argument_type("get_weather", "city", self.tools), "string")
        self.assertEqual(
            get_argument_type("get_weather", "days", self.tools), "integer"
        )
        self.assertIsNone(get_argument_type("get_weather", "missing", self.tools))
        self.assertIsNone(get_argument_type("missing", "city", self.tools))

    def test_parse_arguments_json_python_literal_and_fallback(self):
        cases = [
            ("3", 3, True),
            ("true", True, True),
            ('{"unit": "c"}', {"unit": "c"}, True),
            ("{'unit': 'c'}", {"unit": "c"}, True),
            ("not a literal", "not a literal", False),
        ]
        for value, expected, success in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_arguments(value), (expected, success))


class TestStep3Detector(CustomTestCase):
    def setUp(self):
        self.detector = Step3Detector()
        self.tools = _make_tools()

    def test_has_tool_call(self):
        self.assertTrue(
            self.detector.has_tool_call(
                _step3_block(_step3_call("get_weather", {"city": "Seattle"}))
            )
        )
        self.assertFalse(self.detector.has_tool_call("No tools are needed."))

    def test_parse_steptml_invoke_with_schema_types(self):
        invoke = (
            '<steptml:invoke name="get_weather">'
            '<steptml:parameter name="city">Seattle</steptml:parameter>'
            '<steptml:parameter name="days">3</steptml:parameter>'
            '<steptml:parameter name="exact">true</steptml:parameter>'
            "</steptml:invoke>"
        )
        name, params = self.detector._parse_steptml_invoke(invoke, self.tools)
        self.assertEqual(name, "get_weather")
        self.assertEqual(params, {"city": "Seattle", "days": 3, "exact": True})

    def test_parse_steptml_invoke_without_tools_uses_generic_parsing(self):
        invoke = (
            '<steptml:invoke name="configure">'
            '<steptml:parameter name="count">2</steptml:parameter>'
            '<steptml:parameter name="enabled">False</steptml:parameter>'
            '<steptml:parameter name="label">plain</steptml:parameter>'
            "</steptml:invoke>"
        )
        name, params = self.detector._parse_steptml_invoke(invoke)
        self.assertEqual(name, "configure")
        self.assertEqual(params, {"count": 2, "enabled": False, "label": "plain"})

    def test_parse_steptml_invoke_malformed(self):
        self.assertEqual(
            self.detector._parse_steptml_invoke(
                '<steptml:invoke name="get_weather">incomplete', self.tools
            ),
            (None, {}),
        )

    def test_detect_and_parse_plain_text(self):
        text = "No tools are needed."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_incomplete_block_as_normal_text(self):
        text = "Before <｜tool_calls_begin｜>" + _step3_call(
            "get_weather", {"city": "Seattle"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_call(self):
        text = _step3_block(
            _step3_call(
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
        text = _step3_block(
            _step3_call("get_weather", {"city": "Seattle"}),
            _step3_call("search", {"query": "pizza"}),
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "search"]
        )
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [{"city": "Seattle"}, {"query": "pizza"}],
        )

    def test_detect_and_parse_non_function_call_ignored(self):
        text = _step3_block(
            _step3_call("get_weather", {"city": "Seattle"}, call_type="action")
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_missing_separator_ignored(self):
        call = (
            "<｜tool_call_begin｜>"
            '<steptml:invoke name="get_weather"></steptml:invoke>'
            "<｜tool_call_end｜>"
        )
        result = self.detector.detect_and_parse(_step3_block(call), self.tools)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_unknown_tool_dropped(self):
        result = self.detector.detect_and_parse(
            _step3_block(_step3_call("missing", {"value": "1"})), self.tools
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_streaming_plain_text(self):
        result = self.detector.parse_streaming_increment(
            "A normal response.", self.tools
        )
        self.assertEqual(result.normal_text, "A normal response.")
        self.assertEqual(result.calls, [])

    def test_streaming_partial_begin_token_waits(self):
        first = self.detector.parse_streaming_increment(
            "Checking. <｜tool_calls_", self.tools
        )
        second = self.detector.parse_streaming_increment("begin｜>", self.tools)

        self.assertEqual(first.normal_text, "")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.normal_text, "Checking. ")
        self.assertEqual(second.calls, [])

    def test_streaming_call_across_chunks(self):
        chunks = [
            "<｜tool_calls_begin｜>",
            (
                "<｜tool_call_begin｜>function<｜tool_sep｜>"
                '<steptml:invoke name="get_weather">'
            ),
            '<steptml:parameter name="city">Seattle</steptml:parameter>',
            (
                '<steptml:parameter name="days">3</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜>"
            ),
            "<｜tool_calls_end｜>Done.",
        ]
        results = [
            self.detector.parse_streaming_increment(chunk, self.tools)
            for chunk in chunks
        ]
        calls = [call for result in results for call in result.calls]

        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        streamed_parameters = "".join(
            call.parameters for call in calls if call.parameters
        )
        self.assertEqual(
            json.loads(streamed_parameters), {"city": "Seattle", "days": 3}
        )
        self.assertEqual(results[-1].normal_text, "Done.")

    def test_streaming_multiple_calls_across_chunks(self):
        chunks = [
            "<｜tool_calls_begin｜>",
            (
                "<｜tool_call_begin｜>function<｜tool_sep｜>"
                '<steptml:invoke name="get_weather">'
            ),
            (
                '<steptml:parameter name="city">Seattle</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜>"
            ),
            (
                "<｜tool_call_begin｜>function<｜tool_sep｜>"
                '<steptml:invoke name="search">'
            ),
            (
                '<steptml:parameter name="query">pizza</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜>"
            ),
            "<｜tool_calls_end｜>",
        ]
        calls = []
        for chunk in chunks:
            calls.extend(
                self.detector.parse_streaming_increment(chunk, self.tools).calls
            )

        self.assertEqual(
            [call.name for call in calls if call.name], ["get_weather", "search"]
        )
        parameters_by_index = {}
        for call in calls:
            if call.parameters:
                parameters_by_index.setdefault(call.tool_index, "")
                parameters_by_index[call.tool_index] += call.parameters
        self.assertEqual(json.loads(parameters_by_index[0]), {"city": "Seattle"})
        self.assertEqual(json.loads(parameters_by_index[1]), {"query": "pizza"})

    def test_streaming_unknown_tool_emits_no_call(self):
        chunks = [
            "<｜tool_calls_begin｜>",
            (
                "<｜tool_call_begin｜>function<｜tool_sep｜>"
                '<steptml:invoke name="missing">'
            ),
            "<｜tool_calls_end｜>",
        ]
        calls = []
        for chunk in chunks:
            calls.extend(
                self.detector.parse_streaming_increment(chunk, self.tools).calls
            )
        self.assertEqual(calls, [])

    def test_text_after_finished_tool_block_is_normal(self):
        self.detector.parse_streaming_increment("<｜tool_calls_begin｜>", self.tools)
        self.detector.parse_streaming_increment("<｜tool_calls_end｜>", self.tools)
        result = self.detector.parse_streaming_increment("Afterward.", self.tools)
        self.assertEqual(result.normal_text, "Afterward.")
        self.assertEqual(result.calls, [])

    def test_parser_registry(self):
        parser = FunctionCallParser(self.tools, "step3")
        self.assertIsInstance(parser.detector, Step3Detector)

    def test_structural_tag_is_not_supported(self):
        self.assertFalse(self.detector.supports_structural_tag())
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    unittest.main()
