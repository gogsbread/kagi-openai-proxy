// responses.go is the OpenAI Responses API surface (POST /v1/responses).
//
// It is stateless, like Chat Completions: every request carries the whole
// conversation in input (coding agents such as Pi send store=false), which is
// normalized into Messages and flattened into one Kagi prompt. Tool calls reuse the
// prompt shim in tools.go and are returned as function_call output items.

package translate

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"kagiopenaiproxy/internal/events"
	"kagiopenaiproxy/internal/kagi"
	"kagiopenaiproxy/internal/logx"
)

// responsesParams are the Responses request fields the proxy reads. Others (store,
// prompt_cache_key, max_output_tokens, include, ...) are logged once and ignored.
var responsesParams = map[string]bool{
	"model": true, "input": true, "instructions": true, "stream": true,
	"tools": true, "tool_choice": true, "reasoning": true, "temperature": true,
}

// ResponsesRequest is a validated Responses request. Model is "" when omitted.
type ResponsesRequest struct {
	Model           string
	Messages        []Message
	Stream          bool
	ReasoningEffort string // reasoning.effort; "" when absent
	Tools           []ToolSpec
}

// logOnce logs msg the first time key is seen.
func logOnce(key, msg string) {
	loggedIgnoredMu.Lock()
	first := !loggedIgnored[key]
	loggedIgnored[key] = true
	loggedIgnoredMu.Unlock()
	if first {
		logx.Info("%s (logged once)", msg)
	}
}

// ParseResponsesRequest validates a decoded Responses body.
func ParseResponsesRequest(body any) (ResponsesRequest, *TurnError) {
	obj, ok := body.(map[string]any)
	if !ok {
		return ResponsesRequest{}, InvalidRequest("request body must be a JSON object")
	}
	// Server-side state is not supported; ignoring it would silently drop history.
	for _, key := range []string{"previous_response_id", "conversation"} {
		if obj[key] != nil {
			return ResponsesRequest{}, InvalidRequest(fmt.Sprintf(
				"'%s' is not supported: this proxy keeps no server-side state, so send the whole conversation in 'input'", key))
		}
	}
	for key := range obj {
		if !responsesParams[key] {
			logOnce("responses:"+key, "ignoring unsupported /v1/responses parameter "+strconv.Quote(key))
		}
	}

	var req ResponsesRequest
	if raw, present := obj["model"]; present && raw != nil {
		model, ok := raw.(string)
		if !ok || strings.TrimSpace(model) == "" {
			return ResponsesRequest{}, InvalidRequest("'model' must be a non-empty string")
		}
		req.Model = strings.TrimSpace(model)
	}
	if raw := obj["temperature"]; raw != nil {
		if _, ok := raw.(json.Number); !ok {
			return ResponsesRequest{}, InvalidRequest("'temperature' must be a number")
		}
	}
	if raw := obj["stream"]; raw != nil {
		stream, ok := raw.(bool)
		if !ok {
			return ResponsesRequest{}, InvalidRequest("'stream' must be a boolean")
		}
		req.Stream = stream
	}
	if raw := obj["reasoning"]; raw != nil {
		reasoning, ok := raw.(map[string]any)
		if !ok {
			return ResponsesRequest{}, InvalidRequest("'reasoning' must be an object")
		}
		if raw := reasoning["effort"]; raw != nil {
			effort, ok := raw.(string)
			if !ok {
				return ResponsesRequest{}, InvalidRequest("'reasoning.effort' must be a string")
			}
			if !kagi.IsReasoningEffort(effort) {
				return ResponsesRequest{}, invalidEffort(fmt.Sprintf("'reasoning.effort' %s is not supported; use one of: %s",
					strconv.Quote(effort), strings.Join(kagi.ReasoningEfforts, ", ")))
			}
			req.ReasoningEffort = effort
		}
	}
	if raw := obj["instructions"]; raw != nil {
		instructions, ok := raw.(string)
		if !ok {
			return ResponsesRequest{}, InvalidRequest("'instructions' must be a string")
		}
		if strings.TrimSpace(instructions) != "" {
			req.Messages = append(req.Messages, Message{Role: RoleSystem, Content: instructions})
		}
	}
	switch input := obj["input"].(type) {
	case string:
		req.Messages = append(req.Messages, Message{Role: RoleUser, Content: input})
	case []any:
		msgs, terr := parseInputItems(input)
		if terr != nil {
			return ResponsesRequest{}, terr
		}
		req.Messages = append(req.Messages, msgs...)
	default:
		return ResponsesRequest{}, InvalidRequest("'input' must be a string or an array of input items")
	}
	hasTurn := false
	for _, m := range req.Messages {
		hasTurn = hasTurn || m.Role != RoleSystem
	}
	if !hasTurn {
		return ResponsesRequest{}, InvalidRequest("'input' must contain at least one non-system item")
	}
	tools, terr := parseResponsesTools(obj)
	if terr != nil {
		return ResponsesRequest{}, terr
	}
	req.Tools = tools
	return req, nil
}

// parseInputItems normalizes input items into Messages. Replayed function_call items
// are folded into the preceding assistant message as "[called tool ...]" lines (the
// transcript shape the Chat Completions path produces), function_call_output items
// become tool results, and reasoning items are dropped.
func parseInputItems(items []any) ([]Message, *TurnError) {
	var out []Message
	for i, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			return nil, InvalidRequest(fmt.Sprintf("input[%d] must be an object", i))
		}
		typ, _ := item["type"].(string)
		switch typ {
		case "", "message":
			name, _ := item["role"].(string)
			role, known := roleMap[name]
			if !known || role == RoleTool {
				return nil, InvalidRequest(fmt.Sprintf("input[%d].role %v is not supported", i, item["role"]))
			}
			content, terr := responsesContent(item["content"], fmt.Sprintf("input[%d].content", i))
			if terr != nil {
				return nil, terr
			}
			out = append(out, Message{Role: role, Content: content})
		case "function_call":
			name, _ := item["name"].(string)
			if strings.TrimSpace(name) == "" {
				return nil, InvalidRequest(fmt.Sprintf("input[%d].name must be a non-empty string", i))
			}
			args, _ := item["arguments"].(string)
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			line := "[called tool " + strconv.Quote(name) + " with arguments " + args + "]"
			if n := len(out); n > 0 && out[n-1].Role == RoleAssistant {
				if strings.TrimSpace(out[n-1].Content) == "" {
					out[n-1].Content = line
				} else {
					out[n-1].Content += "\n" + line
				}
			} else {
				out = append(out, Message{Role: RoleAssistant, Content: line})
			}
		case "function_call_output":
			content, terr := responsesContent(item["output"], fmt.Sprintf("input[%d].output", i))
			if terr != nil {
				return nil, terr
			}
			out = append(out, Message{Role: RoleTool, Content: content})
		case "reasoning":
			// Kagi's thinking is not replayed.
		default:
			logOnce("responses-item:"+typ, "ignoring unsupported /v1/responses input item type "+strconv.Quote(typ))
		}
	}
	return out, nil
}

// responsesContent turns a string, an array of content parts, or null into plain text.
func responsesContent(content any, field string) (string, *TurnError) {
	switch c := content.(type) {
	case nil:
		return "", nil
	case string:
		return c, nil
	case []any:
		var pieces []string
		for _, raw := range c {
			part, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			switch part["type"] {
			case "input_text", "output_text", "text":
				if text, ok := part["text"].(string); ok {
					pieces = append(pieces, text)
				}
			case "refusal":
				if text, ok := part["refusal"].(string); ok {
					pieces = append(pieces, text)
				}
			case "input_image", "image_url":
				pieces = append(pieces, ImagePlaceholder)
			case "input_file":
				pieces = append(pieces, "[file omitted]")
			}
		}
		return strings.Join(pieces, "\n"), nil
	}
	return "", InvalidRequest(field + " must be a string or an array of content parts")
}

// parseResponsesTools reads function tools in the Responses shape ({"type":"function",
// "name":...}) or the Chat shape ({"type":"function","function":{...}}). Built-in
// tools (web_search, file_search, ...) are skipped. nil when tool_choice is "none".
func parseResponsesTools(obj map[string]any) ([]ToolSpec, *TurnError) {
	raw := obj["tools"]
	if raw == nil {
		return nil, nil
	}
	if choice, _ := obj["tool_choice"].(string); choice == "none" {
		return nil, nil
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil, InvalidRequest("'tools' must be an array")
	}
	var out []ToolSpec
	for i, item := range arr {
		entry, ok := item.(map[string]any)
		if !ok {
			return nil, InvalidRequest(fmt.Sprintf("tools[%d] must be an object", i))
		}
		if typ, _ := entry["type"].(string); typ != "function" {
			logOnce("responses-tool:"+typ, "ignoring unsupported /v1/responses tool type "+strconv.Quote(typ))
			continue
		}
		fn := entry
		if nested, ok := entry["function"].(map[string]any); ok {
			fn = nested
		}
		name, _ := fn["name"].(string)
		if strings.TrimSpace(name) == "" {
			return nil, InvalidRequest(fmt.Sprintf("tools[%d].name must be a non-empty string", i))
		}
		desc, _ := fn["description"].(string)
		out = append(out, ToolSpec{Name: name, Description: desc, Parameters: fn["parameters"]})
	}
	return out, nil
}

// ResponsesKeepalive is an SSE comment line; SSE clients skip it.
const ResponsesKeepalive = ": keepalive\n\n"

// ResponseStream builds the output of one response, as SSE events or as the final
// response object. Items come in order: at most one assistant message, then any
// function calls.
type ResponseStream struct {
	ID        string
	CreatedAt int64
	Model     string

	seq      int
	output   []any
	msgID    string
	msgIndex int
	msgText  strings.Builder
	msgOpen  bool
}

// NewResponseStream returns an empty stream for one response.
func NewResponseStream(id string, createdAt int64, model string) *ResponseStream {
	return &ResponseStream{ID: id, CreatedAt: createdAt, Model: model}
}

func newItemID(prefix string) string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

// event renders one SSE event; each carries its type and a sequence number.
func (s *ResponseStream) event(name string, payload map[string]any) string {
	payload["type"] = name
	payload["sequence_number"] = s.seq
	s.seq++
	return "event: " + name + "\ndata: " + string(Marshal(payload)) + "\n\n"
}

func (s *ResponseStream) response(status string, u *events.Usage, failure map[string]any) map[string]any {
	output := s.output
	if output == nil {
		output = []any{}
	}
	return map[string]any{
		"id": s.ID, "object": "response", "created_at": s.CreatedAt, "status": status,
		"model": s.Model, "output": output, "error": failure, "incomplete_details": nil,
		"usage": responsesUsage(u),
	}
}

func responsesUsage(u *events.Usage) any {
	if u == nil {
		return nil
	}
	return map[string]any{
		"input_tokens": u.InputTokens, "output_tokens": u.OutputTokens,
		"total_tokens":          u.InputTokens + u.OutputTokens,
		"input_tokens_details":  map[string]any{"cached_tokens": 0},
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
	}
}

func outputText(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
}

func (s *ResponseStream) messageItem(status string, content []any) map[string]any {
	if content == nil {
		content = []any{}
	}
	return map[string]any{"id": s.msgID, "type": "message", "status": status, "role": "assistant", "content": content}
}

// Start is the opening events: response.created and response.in_progress.
func (s *ResponseStream) Start() string {
	return s.event("response.created", map[string]any{"response": s.response("in_progress", nil, nil)}) +
		s.event("response.in_progress", map[string]any{"response": s.response("in_progress", nil, nil)})
}

// TextOpen reports whether the assistant message item is open.
func (s *ResponseStream) TextOpen() bool { return s.msgOpen }

// OpenText starts the assistant message item.
func (s *ResponseStream) OpenText() string {
	s.msgID, s.msgIndex, s.msgOpen = s.itemID("msg_"), len(s.output), true
	s.msgText.Reset()
	return s.event("response.output_item.added", map[string]any{"output_index": s.msgIndex, "item": s.messageItem("in_progress", nil)}) +
		s.event("response.content_part.added", map[string]any{"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0, "part": outputText("")})
}

// TextDelta appends answer text to the open message.
func (s *ResponseStream) TextDelta(text string) string {
	s.msgText.WriteString(text)
	return s.event("response.output_text.delta", map[string]any{
		"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0, "delta": text, "logprobs": []any{}})
}

// CloseText ends the open message and adds it to the output.
func (s *ResponseStream) CloseText() string {
	text := s.msgText.String()
	item := s.messageItem("completed", []any{outputText(text)})
	s.output = append(s.output, item)
	s.msgOpen = false
	return s.event("response.output_text.done", map[string]any{
		"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0, "text": text, "logprobs": []any{}}) +
		s.event("response.content_part.done", map[string]any{
			"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0, "part": outputText(text)}) +
		s.event("response.output_item.done", map[string]any{"output_index": s.msgIndex, "item": item})
}

// FunctionCall adds one complete function_call item. call_id is the shim's tool call
// id; the client sends it back with the matching function_call_output.
func (s *ResponseStream) FunctionCall(c ToolCall) string {
	index, itemID := len(s.output), newItemID("fc_")
	c.ID = s.callID(c, index)
	item := func(status, args string) map[string]any {
		return map[string]any{"id": itemID, "type": "function_call", "status": status,
			"call_id": c.ID, "name": c.Function.Name, "arguments": args}
	}
	done := item("completed", c.Function.Arguments)
	s.output = append(s.output, done)
	return s.event("response.output_item.added", map[string]any{"output_index": index, "item": item("in_progress", "")}) +
		s.event("response.function_call_arguments.delta", map[string]any{"item_id": itemID, "output_index": index, "delta": c.Function.Arguments}) +
		s.event("response.function_call_arguments.done", map[string]any{
			"item_id": itemID, "output_index": index, "name": c.Function.Name, "arguments": c.Function.Arguments}) +
		s.event("response.output_item.done", map[string]any{"output_index": index, "item": done})
}

// Completed is the last event of a successful stream.
func (s *ResponseStream) Completed(u *events.Usage) string {
	return s.event("response.completed", map[string]any{"response": s.response("completed", u, nil)})
}

// Body is the non-streaming response object.
func (s *ResponseStream) Body(u *events.Usage) any { return s.response("completed", u, nil) }

// Failed ends a stream with an error event and response.failed.
func (s *ResponseStream) Failed(e *TurnError) string {
	failure := map[string]any{"code": e.code, "message": e.message}
	return s.event("error", map[string]any{"code": e.code, "message": e.message, "param": nil}) +
		s.event("response.failed", map[string]any{"response": s.response("failed", nil, failure)})
}
