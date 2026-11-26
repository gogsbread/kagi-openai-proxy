package translate

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func mustParseResponses(t *testing.T, raw string) ResponsesRequest {
	t.Helper()
	body, err := DecodeBody([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	req, terr := ParseResponsesRequest(body)
	if terr != nil {
		t.Fatalf("ParseResponsesRequest: %v", terr)
	}
	return req
}

// The first request Pi 1.6.1 sends, with the system prompt and tools trimmed.
func TestParseResponsesRequestPiFirstTurn(t *testing.T) {
	req := mustParseResponses(t, `{"model":"glm-5-3-flash:offline:low","input":[
		{"role":"system","content":"You are a coding assistant."},
		{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"stream":true,"prompt_cache_key":"s1","store":false,"max_output_tokens":16384,
		"tools":[{"type":"function","name":"bash","description":"Execute a bash command",
		"parameters":{"type":"object","required":["command"],"properties":{"command":{"type":"string"}}}}]}`)
	if req.Model != "glm-5-3-flash:offline:low" || !req.Stream {
		t.Fatalf("model/stream = %q/%v", req.Model, req.Stream)
	}
	want := []Message{{RoleSystem, "You are a coding assistant."}, {RoleUser, "hi"}}
	if !slices.Equal(req.Messages, want) {
		t.Fatalf("messages = %+v", req.Messages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "bash" || req.Tools[0].Parameters == nil {
		t.Fatalf("tools = %+v", req.Tools)
	}
}

func TestParseResponsesRequestToolReplay(t *testing.T) {
	req := mustParseResponses(t, `{"model":"m","instructions":"Be brief.","input":[
		{"role":"user","content":"list files"},
		{"type":"reasoning","id":"rs_1","summary":[]},
		{"type":"message","role":"assistant","id":"msg_1","status":"completed",
		 "content":[{"type":"output_text","text":"Let me look.","annotations":[]}]},
		{"type":"function_call","id":"fc_1","call_id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}"},
		{"type":"function_call_output","call_id":"call_1","output":"a.go"},
		{"type":"function_call","call_id":"call_2","name":"read","arguments":""},
		{"type":"function_call_output","call_id":"call_2","output":[{"type":"input_text","text":"package a"}]}]}`)
	want := []Message{
		{RoleSystem, "Be brief."},
		{RoleUser, "list files"},
		{RoleAssistant, "Let me look.\n[called tool \"bash\" with arguments {\"command\":\"ls\"}]"},
		{RoleTool, "a.go"},
		{RoleAssistant, "[called tool \"read\" with arguments {}]"},
		{RoleTool, "package a"},
	}
	if !slices.Equal(req.Messages, want) {
		t.Fatalf("messages = %+v", req.Messages)
	}
}

func TestParseResponsesRequestToolsAndRejects(t *testing.T) {
	req := mustParseResponses(t, `{"input":"hi","tools":[{"type":"web_search"},{"type":"function","function":{"name":"calc"}}]}`)
	if len(req.Tools) != 1 || req.Tools[0].Name != "calc" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	req = mustParseResponses(t, `{"input":"hi","tool_choice":"none","tools":[{"type":"function","name":"calc"}]}`)
	if req.Tools != nil {
		t.Fatalf("tool_choice none must drop tools: %+v", req.Tools)
	}
	for _, raw := range []string{
		`{"input":[{"role":"system","content":"x"}]}`,
		`{"input":"hi","previous_response_id":"resp_1"}`,
		`{"model":"m"}`,
	} {
		body, _ := DecodeBody([]byte(raw))
		if _, terr := ParseResponsesRequest(body); terr == nil {
			t.Fatalf("%s: expected an error", raw)
		}
	}
}

func responseEvents(t *testing.T, frames string) ([]string, []map[string]any) {
	t.Helper()
	var names []string
	var data []map[string]any
	for _, block := range strings.Split(strings.TrimSpace(frames), "\n\n") {
		name, payload, ok := strings.Cut(block, "\ndata: ")
		if !ok || !strings.HasPrefix(name, "event: ") {
			t.Fatalf("bad frame %q", block)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(payload), &m); err != nil {
			t.Fatal(err)
		}
		names = append(names, strings.TrimPrefix(name, "event: "))
		data = append(data, m)
	}
	return names, data
}

func TestResponseStreamEvents(t *testing.T) {
	s := NewResponseStream("resp_1", 1, "m")
	frames := s.Start() + s.OpenText() + s.TextDelta("he") + s.TextDelta("llo") + s.CloseText() +
		s.FunctionCall(ToolCall{ID: "call_1", Type: "function", Function: functionJSON{Name: "bash", Arguments: `{"command":"ls"}`}}) +
		s.Completed(nil)
	names, data := responseEvents(t, frames)
	want := []string{"response.created", "response.in_progress",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done", "response.output_item.done",
		"response.output_item.added", "response.function_call_arguments.delta",
		"response.function_call_arguments.done", "response.output_item.done",
		"response.completed"}
	if !slices.Equal(names, want) {
		t.Fatalf("events = %v", names)
	}
	for i, d := range data {
		if d["type"] != names[i] || d["sequence_number"] != float64(i) {
			t.Fatalf("event %d: type %v, sequence_number %v", i, d["type"], d["sequence_number"])
		}
	}
	out := data[len(data)-1]["response"].(map[string]any)["output"].([]any)
	msg, call := out[0].(map[string]any), out[1].(map[string]any)
	text := msg["content"].([]any)[0].(map[string]any)["text"]
	if msg["type"] != "message" || text != "hello" || call["call_id"] != "call_1" || call["arguments"] != `{"command":"ls"}` {
		t.Fatalf("output = %v", out)
	}
}
