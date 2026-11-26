// responses_thread.go lets /v1/responses ids carry the Kagi thread.
//
// The ids the proxy issues (resp_, msg_, call_) embed the Kagi branch uuid and a hash
// of the tool set the branch was last told about. A client that replays those ids on
// its next request continues that branch: only the items after the last tagged
// assistant reply are sent. The history before it is not checked, so edits, rewinds
// and forks stay on the same branch.

package translate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

var (
	uuidPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	// optional numeric suffixes: the per-response nonce, then the call index
	tagPattern = regexp.MustCompile(`^(?:resp|msg|call)_([0-9a-f]{32})_([0-9a-f]{16})(?:_[0-9]+){0,2}$`)
)

// ResponseTag is the id suffix "<32 hex branch>_<16 hex tools hash>", or "" when the
// branch is not a lowercase uuid (the reply then gets random ids and cannot be continued).
func ResponseTag(branchUUID, hash string) string {
	if len(hash) != 16 || !uuidPattern.MatchString(branchUUID) {
		return ""
	}
	return strings.ReplaceAll(branchUUID, "-", "") + "_" + hash
}

func expandUUID(h string) string {
	return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

// tag is the suffix of the response id when it carries a thread tag.
func (s *ResponseStream) tag() string {
	if strings.HasPrefix(s.ID, "resp_") && tagPattern.MatchString(s.ID) {
		return strings.TrimPrefix(s.ID, "resp_")
	}
	return ""
}

// itemID is the id of the assistant message: tagged when the response id is.
func (s *ResponseStream) itemID(prefix string) string {
	if t := s.tag(); t != "" {
		return prefix + t
	}
	return newItemID(prefix)
}

// callID is the call_id of a function call: tagged when the response id is, else the
// shim's own id. It stays within [a-zA-Z0-9_-] and 64 characters, which Pi requires.
func (s *ResponseStream) callID(c ToolCall, index int) string {
	if t := s.tag(); t != "" {
		return "call_" + t + "_" + strconv.Itoa(index)
	}
	return c.ID
}

// ToolsHash identifies a tool set by its names, descriptions and schemas. Order does
// not matter, and no tools hashes the same as an empty list.
func ToolsHash(tools []ToolSpec) string {
	sorted := make([]ToolSpec, len(tools))
	copy(sorted, tools)
	slices.SortStableFunc(sorted, func(a, b ToolSpec) int { return strings.Compare(a.Name, b.Name) })
	data, _ := json.Marshal(sorted)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])[:16]
}

// Continuation is the part of a request that the Kagi branch has not seen yet.
type Continuation struct {
	Branch    string    // Kagi branch uuid to continue
	ToolsHash string    // tool set the branch was last told about
	Delta     []Message // the items after the last tagged assistant reply
}

// assistantTag reads the thread tag from an assistant message or function_call item.
func assistantTag(raw any) (branch, hash string, ok bool) {
	item, isMap := raw.(map[string]any)
	if !isMap {
		return
	}
	var id string
	switch typ, _ := item["type"].(string); {
	case typ == "function_call":
		id, _ = item["call_id"].(string)
	case typ == "" || typ == "message":
		if role, _ := item["role"].(string); role == "assistant" {
			id, _ = item["id"].(string)
		}
	}
	m := tagPattern.FindStringSubmatch(id)
	if m == nil {
		return
	}
	return m[1], m[2], true
}

// FindContinuation finds the last assistant reply carrying a proxy tag. Everything after
// it (tool results, new user messages, replies from other models) is the delta for that
// branch. A request with no tagged reply, or nothing after it, starts a new thread.
func FindContinuation(body any) (Continuation, bool) {
	obj, _ := body.(map[string]any)
	items, _ := obj["input"].([]any)
	for i := len(items) - 1; i >= 0; i-- {
		branch, hash, ok := assistantTag(items[i])
		if !ok {
			continue
		}
		delta, terr := parseInputItems(items[i+1:])
		if terr != nil || len(delta) == 0 {
			return Continuation{}, false
		}
		return Continuation{Branch: expandUUID(branch), ToolsHash: hash, Delta: delta}, true
	}
	return Continuation{}, false
}
