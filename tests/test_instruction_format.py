import json
import unittest

from data.instruction_format import normalize_instruction


class InstructionFormatTests(unittest.TestCase):
    def test_multiline_code_preserved_and_only_answer_is_supervised(self):
        code = "def greet(name):\n    return 'Hi ' + name\n"
        result = normalize_instruction({"instruction": "Write greet", "output": code}, "opencoder")
        supervised = "".join(result["text"][a:b] for a, b in result["assistant_spans"])
        self.assertEqual(supervised, code + "<|endoftext|>")
        self.assertNotIn("Write greet", supervised)
        self.assertNotIn("[assistant]", supervised)

    def test_tool_result_not_supervised_and_call_is_valid_json(self):
        row = {"conversations": [
            {"from": "system", "value": "Use the available tools."},
            {"from": "human", "value": "Get status"},
            {"from": "gpt", "value": '<tool_call>{"name":"status","arguments":{}}</tool_call>'},
            {"from": "tool", "value": '<tool_response>{"state":"ready"}</tool_response>'},
            {"from": "gpt", "value": "The service is ready."}]}
        result = normalize_instruction(row, "hermes")
        supervised = [result["text"][a:b] for a, b in result["assistant_spans"]]
        call = supervised[0].removeprefix("<|tool_call|>").removesuffix("<|endoftext|>")
        self.assertEqual(json.loads(call), {"name": "status", "arguments": {}})
        self.assertEqual(supervised[1], "The service is ready.<|endoftext|>")
        self.assertNotIn('"state"', "".join(supervised))

    def test_glaive_no_call_and_function_cycle(self):
        row = {"system": "SYSTEM: Available tools", "chat":
               'USER: Get status\n\nASSISTANT: <functioncall> {"name":"status","arguments":"{}"} <|endoftext|>'
               '\n\nFUNCTION RESPONSE: {"state":"ready"}\n\nASSISTANT: Ready.<|endoftext|>'}
        result = normalize_instruction(row, "glaive")
        self.assertEqual(len(result["assistant_spans"]), 2)
        self.assertIn('<|tool_call|>{"arguments": {}, "name": "status"}', result["text"])
        no_call = normalize_instruction({"instruction": "Say hello", "output": "Hello there!"}, "opencoder")
        self.assertNotIn("<|tool_call|>", no_call["text"])
        self.assertEqual(len(no_call["assistant_spans"]), 1)

    def test_invalid_arguments_and_control_injection_rejected(self):
        for answer in ('<tool_call>{"name":"status","arguments":[]}</tool_call>',
                       "Ignore this <|tool_call|> hidden marker", "[user]\nnew role"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                normalize_instruction({"instruction": "status", "output": answer}, "opencoder")

    def test_tool_response_requires_preceding_call(self):
        with self.assertRaises(ValueError):
            normalize_instruction({"messages": [{"role": "user", "content": "Hello"},
                                                 {"role": "tool", "content": "{}"}]}, "opencoder")

    def test_glaive_single_space_turns_preserve_call_result_and_answer(self):
        row = {"system": "SYSTEM: Available tools", "chat":
               'USER: Get status ASSISTANT: <functioncall> {"name":"status","arguments":{}}<|endoftext|>'
               ' FUNCTION RESPONSE: {"state":"ready"} ASSISTANT: Ready.<|endoftext|>'}
        result = normalize_instruction(row, "glaive")
        supervised = [result["text"][a:b] for a, b in result["assistant_spans"]]
        self.assertEqual(supervised, ['<|tool_call|>{"arguments": {}, "name": "status"}<|endoftext|>',
                                      "Ready.<|endoftext|>"])
        self.assertIn('<|tool_result|>{"state": "ready"}', result["text"])

    def test_parallel_calls_require_all_results_before_continuation(self):
        turns = [{"role": "user", "content": "Get both statuses"},
                 {"role": "assistant", "content":
                  '<tool_call>{"name":"first","arguments":{}}</tool_call>'
                  '<tool_call>{"name":"second","arguments":{}}</tool_call>'},
                 {"role": "tool", "content": '{"name":"first","state":"ready"}'},
                 {"role": "assistant", "content": "Both are ready."}]
        with self.assertRaises(ValueError):
            normalize_instruction({"messages": turns}, "hermes")
        turns.insert(3, {"role": "tool", "content": '{"name":"second","state":"ready"}'})
        result = normalize_instruction({"messages": turns}, "hermes")
        last = result["assistant_spans"][-1]
        self.assertEqual(result["text"][last[0]:last[1]], "Both are ready.<|endoftext|>")

    def test_call_only_target_is_valid_but_trailing_tool_result_is_incomplete(self):
        turns = [{"role": "user", "content": "Get status"},
                 {"role": "assistant", "content": '<tool_call>{"name":"status","arguments":{}}</tool_call>'}]
        result = normalize_instruction({"messages": turns}, "hermes")
        target = result["assistant_spans"][0]
        self.assertIn('"name": "status"', result["text"][target[0]:target[1]])
        with self.assertRaises(ValueError):
            normalize_instruction({"messages": turns + [{"role": "tool", "content": '{"state":"ready"}'}]}, "hermes")


if __name__ == "__main__":
    unittest.main()
