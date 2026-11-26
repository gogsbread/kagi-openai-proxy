// responses_threads.go continues one Kagi thread across /v1/responses requests.
//
// Clients such as Pi resend the whole conversation but echo back the ids the proxy
// issued. Those ids carry the Kagi branch uuid and a hash of the tool set. When a
// request replays a tagged assistant reply, only the items after it are sent to that
// branch. Every follow-up names the current tools; when the tool set changed it also
// carries the full new tool block, because a Kagi thread cannot re-render its first
// message. Requests without a tagged reply start a new thread. The proxy keeps no state.

package server

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"kagiopenaiproxy/internal/kagi"
	"kagiopenaiproxy/internal/logx"
	"kagiopenaiproxy/internal/translate"
)

// responseThread is the thread decision for one /v1/responses request.
type responseThread struct {
	cont         *translate.Continuation // nil = start a new thread
	followUp     string                  // prompt for the follow-up turn
	hash         string                  // this request's tool set hash; goes into the reply ids
	toolsChanged bool                    // the follow-up carries the full tool block
}

func planResponseThread(body any, req translate.ResponsesRequest) responseThread {
	t := responseThread{hash: translate.ToolsHash(req.Tools)}
	if c, ok := translate.FindContinuation(body); ok {
		t.cont = &c
		t.toolsChanged = c.ToolsHash != t.hash
		t.followUp = followUpPrompt(c.Delta, req.Tools, t.toolsChanged)
	}
	return t
}

// responseID is the id of this reply: it carries the branch and tools hash when the
// branch can be encoded, and is random otherwise (the next request then starts a new thread).
// The trailing nonce keeps ids unique across turns of the same thread; msg_ and call_ ids
// inherit it.
func (t responseThread) responseID(branch string) string {
	if tag := translate.ResponseTag(branch, t.hash); tag != "" {
		return fmt.Sprintf("resp_%s_%06d", tag, time.Now().UnixNano()%1_000_000)
	}
	return newResponseID()
}

// threadIsSaved keeps reusable threads alive: under the delete lifecycle they are
// created as Temporary instead, as /v1/ask does.
func threadIsSaved(saved *bool, deletes bool) *bool {
	if deletes {
		temporary := false
		return &temporary
	}
	return saved
}

// createOrContinue sends only the new items to the planned branch, or starts a new
// thread with the full flattened prompt. A follow-up Kagi rejects falls back to a new thread.
func (p *Proxy) createOrContinue(ctx context.Context, t responseThread, content, modelID string, opts kagi.TurnOptions) (kagi.CreatedTurn, error) {
	if t.cont != nil {
		created, err := p.kagi.Respond(ctx, t.cont.Branch, t.followUp, modelID, opts)
		if err == nil {
			if created.BranchUUID == "" {
				created.BranchUUID = t.cont.Branch
			}
			logx.Info("responses follow-up in Kagi thread %s (tools changed: %v): sent %d chars instead of %d",
				created.ConversationUUID, t.toolsChanged, len(t.followUp), len(content))
			return created, nil
		}
		var kerr *kagi.Error
		if !errors.As(err, &kerr) || !fallBackToNewThread(kerr.Kind) || ctx.Err() != nil {
			return created, err
		}
		logx.Warn("follow-up in Kagi branch %s failed (%s); starting a new thread", t.cont.Branch, kerr.Message)
	}
	return p.kagi.CreateTurn(ctx, content, modelID, opts)
}

func fallBackToNewThread(kind kagi.Kind) bool {
	return kind == kagi.KindBadRequest || kind == kagi.KindUpstream || kind == kagi.KindProtocol
}

// followUpPrompt renders the new items of a follow-up turn, then the tool note. A lone
// user message is sent as-is; tool results keep the "Tool result:" label the flattened
// transcript uses.
func followUpPrompt(delta []translate.Message, tools []translate.ToolSpec, toolsChanged bool) string {
	var parts []string
	for _, m := range delta {
		text := strings.TrimSpace(m.Content)
		switch {
		case m.Role == translate.RoleSystem:
			parts = append(parts, "[System instructions]\n"+text)
		case m.Role == translate.RoleTool:
			parts = append(parts, "Tool result: "+text)
		case m.Role == translate.RoleAssistant:
			parts = append(parts, "Assistant: "+text)
		case len(delta) > 1:
			parts = append(parts, "User: "+text)
		default:
			parts = append(parts, text)
		}
	}
	if note := toolNote(tools, toolsChanged); note != "" {
		parts = append(parts, note)
	}
	return strings.Join(parts, "\n\n")
}

// toolNote tells the thread which tools exist now. An unchanged set gets a one-line
// reminder with the tool names; a changed set gets the full block, which replaces the
// one in the thread's first message.
func toolNote(tools []translate.ToolSpec, changed bool) string {
	switch {
	case len(tools) == 0 && changed:
		return "[Tools updated: no tools are available now. Answer normally and never write TOOL_CALL.]"
	case len(tools) == 0:
		return ""
	case changed:
		return "[Tools updated: the list below replaces every tool list earlier in this conversation.]\n" +
			translate.ToolPrompt(tools)
	}
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	return "[Tools available: " + strings.Join(names, ", ") +
		". To call one, reply with only TOOL_CALL:{\"name\":...,\"arguments\":{...}} line(s); otherwise answer normally.]"
}
