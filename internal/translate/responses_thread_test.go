package translate

import (
	"slices"
	"testing"
)

func continuationOf(t *testing.T, raw string) (Continuation, bool) {
	t.Helper()
	body, err := DecodeBody([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return FindContinuation(body)
}

func TestResponsesContinuation(t *testing.T) {
	const branch = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
	const tools = `"tools":[{"type":"function","name":"bash","parameters":{}}]`
	first := `{"model":"m","input":[{"role":"system","content":"sys"},{"role":"user","content":"list files"}],` + tools + `}`
	if _, ok := continuationOf(t, first); ok {
		t.Fatal("a first request must start a new thread")
	}

	req := mustParseResponses(t, first)
	rs := NewResponseStream("resp_"+ResponseTag(branch, ToolsHash(req.Tools)), 1, "m")
	rs.OpenText()
	rs.TextDelta("Let me look.")
	rs.CloseText()
	rs.FunctionCall(ToolCall{ID: "call_x", Type: "function", Function: functionJSON{Name: "bash", Arguments: `{"command":"ls"}`}})
	msgID := rs.output[0].(map[string]any)["id"].(string)
	callID := rs.output[1].(map[string]any)["call_id"].(string)

	second := func(user, tail string) string {
		return `{"model":"other","input":[{"role":"system","content":"sys"},
			{"role":"user","content":[{"type":"input_text","text":"` + user + `"}]},
			{"type":"message","role":"assistant","id":"` + msgID + `","status":"completed",
			 "content":[{"type":"output_text","text":"Let me look.","annotations":[]}]},
			{"type":"function_call","id":"fc_1","call_id":"` + callID + `","name":"bash","arguments":"{\"command\":\"ls\"}"},
			{"type":"function_call_output","call_id":"` + callID + `","output":"a.go"}` + tail + `],` + tools + `}`
	}
	cont, ok := continuationOf(t, second("list files", ""))
	if !ok || cont.Branch != branch || cont.ToolsHash != ToolsHash(req.Tools) {
		t.Fatalf("expected a continuation of %s, got %+v %v", branch, cont, ok)
	}
	if want := []Message{{Role: RoleTool, Content: "a.go"}}; !slices.Equal(cont.Delta, want) {
		t.Fatalf("delta = %+v", cont.Delta)
	}
	if _, ok := continuationOf(t, second("list all files", "")); !ok {
		t.Fatal("an edited earlier message must keep the branch")
	}
	other := `,{"type":"message","role":"assistant","content":[{"type":"output_text","text":"from another model"}]},
		{"role":"user","content":"go on"}`
	cont, ok = continuationOf(t, second("list files", other))
	if !ok || len(cont.Delta) != 3 || cont.Delta[1].Role != RoleAssistant || cont.Delta[2].Content != "go on" {
		t.Fatalf("untagged replies must be part of the delta, got %+v %v", cont.Delta, ok)
	}
	ends := `{"model":"m","input":[{"role":"user","content":"hi"},
		{"type":"message","role":"assistant","id":"` + msgID + `","content":[{"type":"output_text","text":"Hello"}]}]}`
	if _, ok := continuationOf(t, ends); ok {
		t.Fatal("nothing after the tagged reply must start a new thread")
	}
}

func TestToolsHash(t *testing.T) {
	a := []ToolSpec{{Name: "read"}, {Name: "bash"}}
	b := []ToolSpec{{Name: "bash"}, {Name: "read"}}
	if ToolsHash(a) != ToolsHash(b) || len(ToolsHash(a)) != 16 {
		t.Fatal("tool order must not change the hash")
	}
	if ToolsHash(nil) != ToolsHash([]ToolSpec{}) {
		t.Fatal("nil and empty tool lists must hash the same")
	}
	if ToolsHash(a) == ToolsHash(a[:1]) || ToolsHash(a) == ToolsHash([]ToolSpec{{Name: "read", Description: "x"}, {Name: "bash"}}) {
		t.Fatal("a changed tool set must change the hash")
	}
}

func TestResponseTag(t *testing.T) {
	if ResponseTag("not-a-uuid", "0123456789abcdef") != "" || ResponseTag("0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b", "short") != "" {
		t.Fatal("invalid branch or hash must give no tag")
	}
	s := NewResponseStream("resp_0190a1b2c3d47e5f8a9b0c1d2e3f4a5b_0123456789abcdef", 1, "m")
	if got := s.callID(ToolCall{ID: "call_x"}, 3); got != "call_0190a1b2c3d47e5f8a9b0c1d2e3f4a5b_0123456789abcdef_3" || len(got) > 64 {
		t.Fatalf("call id = %q", got)
	}
	n := NewResponseStream("resp_0190a1b2c3d47e5f8a9b0c1d2e3f4a5b_0123456789abcdef_123456", 1, "m")
	got := n.callID(ToolCall{ID: "call_x"}, 9)
	if got != "call_0190a1b2c3d47e5f8a9b0c1d2e3f4a5b_0123456789abcdef_123456_9" || len(got) > 64 {
		t.Fatalf("call id with nonce = %q", got)
	}
	branch, hash, ok := assistantTag(map[string]any{"type": "function_call", "call_id": got})
	if !ok || branch != "0190a1b2c3d47e5f8a9b0c1d2e3f4a5b" || hash != "0123456789abcdef" {
		t.Fatalf("call id with nonce must still parse, got %q %q %v", branch, hash, ok)
	}
}
