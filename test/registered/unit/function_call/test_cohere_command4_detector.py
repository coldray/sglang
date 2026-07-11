"""Unit tests for CohereCommand4Detector - no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
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


def _cohere_block(calls, prefix="", suffix=""):
    return prefix + "<|START_ACTION|>" + json.dumps(calls) + "<|END_ACTION|>" + suffix


def _cohere_call(name, parameters, call_id="0"):
    return {
        "tool_call_id": call_id,
        "tool_name": name,
        "parameters": parameters,
    }


class TestCohereCommand4Detector(CustomTestCase):
    def setUp(self):
        self.detector = CohereCommand4Detector()
        self.tools = _make_tools()

    def test_has_tool_call(self):
        self.assertTrue(
            self.detector.has_tool_call(
                _cohere_block([_cohere_call("get_weather", {"city": "Seattle"})])
            )
        )
        self.assertFalse(self.detector.has_tool_call("No tools are needed."))

    def test_normalize_calls(self):
        normalized = self.detector._normalize_calls(
            _cohere_call("get_weather", {"city": "Seattle"})
        )
        self.assertEqual(
            normalized,
            [{"name": "get_weather", "parameters": {"city": "Seattle"}}],
        )

        already_normalized = {"name": "search", "parameters": {"query": "pizza"}}
        self.assertEqual(
            self.detector._normalize_calls([None, "invalid", already_normalized]),
            [already_normalized],
        )
        self.assertEqual(self.detector._normalize_calls("invalid"), [])

    def test_detect_and_parse_plain_text(self):
        text = "No tools are needed."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_call(self):
        text = _cohere_block(
            [_cohere_call("get_weather", {"city": "Seattle", "days": 3})],
            prefix="Checking. ",
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking. ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Seattle", "days": 3}
        )

    def test_detect_and_parse_multiple_calls(self):
        text = _cohere_block(
            [
                _cohere_call("get_weather", {"city": "Seattle"}, "0"),
                _cohere_call("search", {"query": "pizza"}, "1"),
            ]
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "search"]
        )
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [{"city": "Seattle"}, {"query": "pizza"}],
        )

    def test_detect_and_parse_native_name_shape(self):
        calls = [{"name": "search", "arguments": {"query": "pizza"}}]
        result = self.detector.detect_and_parse(_cohere_block(calls), self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "pizza"})

    def test_detect_and_parse_unknown_tool_dropped(self):
        text = _cohere_block([_cohere_call("missing", {"value": 1})])
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_complete_json_without_end_token(self):
        text = "<|START_ACTION|>" + json.dumps(
            [_cohere_call("search", {"query": "pizza"})]
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    def test_detect_and_parse_truncated_json(self):
        text = (
            "<|START_ACTION|>"
            '[{"tool_call_id":"0","tool_name":"search",'
            '"parameters":{"query":"pizza"}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "pizza"})

    def test_detect_and_parse_malformed_json(self):
        result = self.detector.detect_and_parse(
            "Before <|START_ACTION|>not-json<|END_ACTION|>", self.tools
        )
        self.assertEqual(result.normal_text, "Before ")
        self.assertEqual(result.calls, [])

    def test_streaming_plain_text(self):
        result = self.detector.parse_streaming_increment(
            "A normal response.", self.tools
        )
        self.assertEqual(result.normal_text, "A normal response.")
        self.assertEqual(result.calls, [])

    def test_streaming_partial_start_token(self):
        first = self.detector.parse_streaming_increment(
            "Checking. <|START_ACT", self.tools
        )
        second = self.detector.parse_streaming_increment("ION|>", self.tools)

        self.assertEqual(first.normal_text, "Checking. ")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.normal_text, "")
        self.assertEqual(second.calls, [])

    def test_streaming_call_across_chunks(self):
        wire = _cohere_block(
            [_cohere_call("get_weather", {"city": "Seattle", "days": 3})],
            prefix="Checking. ",
            suffix="Done.",
        )
        split_at = wire.index("<|START_ACTION|>") + len("<|START_ACTION|>") + 20
        chunks = [wire[:split_at], wire[split_at:]]

        first = self.detector.parse_streaming_increment(chunks[0], self.tools)
        second = self.detector.parse_streaming_increment(chunks[1], self.tools)
        third = self.detector.parse_streaming_increment("", self.tools)

        self.assertEqual(first.normal_text, "Checking. ")
        self.assertEqual(first.calls, [])
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(second.calls[0].parameters), {"city": "Seattle", "days": 3}
        )
        self.assertEqual(third.normal_text, "Done.")

    def test_streaming_multiple_calls_in_one_block(self):
        wire = _cohere_block(
            [
                _cohere_call("get_weather", {"city": "Seattle"}, "0"),
                _cohere_call("search", {"query": "pizza"}, "1"),
            ]
        )
        first = self.detector.parse_streaming_increment(wire, self.tools)
        second = self.detector.parse_streaming_increment("", self.tools)
        calls = first.calls + second.calls
        self.assertEqual([call.name for call in calls], ["get_weather", "search"])

    def test_parser_registry(self):
        parser = FunctionCallParser(self.tools, "cohere_command4")
        self.assertIsInstance(parser.detector, CohereCommand4Detector)

    def test_structure_info(self):
        self.assertFalse(self.detector.supports_structural_tag())
        info = self.detector.structure_info()("get_weather")
        self.assertEqual(info.trigger, "<|START_ACTION|>")
        self.assertIn('"tool_name": "get_weather"', info.begin)
        self.assertEqual(info.end, "}]<|END_ACTION|>")


if __name__ == "__main__":
    unittest.main()
