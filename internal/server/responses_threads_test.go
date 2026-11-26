package server

import (
	"strings"
	"testing"

	"kagiopenaiproxy/internal/translate"
)

func TestFollowUpPrompt(t *testing.T) {
	bash := []translate.ToolSpec{{Name: "bash", Parameters: map[string]any{"type": "object"}}}
	user := []translate.Message{{Role: translate.RoleUser, Content: "and now?"}}
	if got := followUpPrompt(user, nil, false); got != "and now?" {
		t.Fatalf("lone user message = %q", got)
	}
	results := []translate.Message{{Role: translate.RoleTool, Content: "a.go"}, {Role: translate.RoleTool, Content: "b.go"}}
	got := followUpPrompt(results, bash, false)
	if !strings.Contains(got, "Tool result: a.go\n\nTool result: b.go") || !strings.Contains(got, "[Tools available: bash.") ||
		strings.Contains(got, "parameters:") {
		t.Fatalf("unchanged tools = %q", got)
	}
	got = followUpPrompt(user, bash, true)
	if !strings.Contains(got, "[Tools updated:") || !strings.Contains(got, "parameters:") {
		t.Fatalf("changed tools = %q", got)
	}
	got = followUpPrompt(user, nil, true)
	if !strings.Contains(got, "no tools are available") {
		t.Fatalf("removed tools = %q", got)
	}
	mixed := []translate.Message{{Role: translate.RoleAssistant, Content: "hi"}, {Role: translate.RoleUser, Content: "go on"}}
	if got := followUpPrompt(mixed, nil, false); got != "Assistant: hi\n\nUser: go on" {
		t.Fatalf("mixed delta = %q", got)
	}
}

func TestResponseThreadID(t *testing.T) {
	th := responseThread{hash: "0123456789abcdef"}
	const prefix = "resp_0190a1b2c3d47e5f8a9b0c1d2e3f4a5b_0123456789abcdef_"
	if got := th.responseID("0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"); !strings.HasPrefix(got, prefix) || len(got) != len(prefix)+6 {
		t.Fatalf("tagged id = %q", got)
	}
	if got := th.responseID("not-a-uuid"); !strings.HasPrefix(got, "resp_") || strings.Contains(got, "_0123") {
		t.Fatalf("untagged id = %q", got)
	}
}
