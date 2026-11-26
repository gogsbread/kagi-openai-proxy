package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"kagiopenaiproxy/internal/events"
	"kagiopenaiproxy/internal/kagi"
	"kagiopenaiproxy/internal/logx"
	"kagiopenaiproxy/internal/translate"
)

// responses serves POST /v1/responses (the OpenAI Responses API). Like
// /v1/chat/completions it is stateless: the whole conversation arrives in input
// (clients such as Pi send store=false) and is flattened into a new Kagi thread.
func (p *Proxy) responses(w http.ResponseWriter, r *http.Request) {
	if !p.requireAuth(w, r) {
		return
	}
	settings := p.settings
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return // client gone while sending the body
	}
	body, err := translate.DecodeBody(raw)
	if err != nil {
		writeError(w, translate.InvalidRequest("request body is not valid JSON"))
		return
	}
	req, terr := translate.ParseResponsesRequest(body)
	if terr != nil {
		writeError(w, terr)
		return
	}
	model, terr := p.resolveResponsesModel(r, req.Model, req.ReasoningEffort)
	if terr != nil {
		writeError(w, terr)
		return
	}
	content := translate.Flatten(translate.WithToolPrompt(req.Messages, req.Tools), model.contextWindow)

	switch p.acquireSlot(r.Context()) {
	case slotClientGone:
		logx.Info("client disconnected while queued for a Kagi turn slot; nothing was sent to Kagi")
		return // the client is gone: no status is written
	case slotTimeout:
		writeError(w, translate.NewTurnError(429, "rate_limit_error", "proxy_busy", fmt.Sprintf(
			"another Kagi turn is in progress; waited %ss for a free slot", strconv.FormatFloat(settings.QueueWait.Seconds(), 'g', -1, 64))))
		return
	}
	opts := kagi.TurnOptions{
		EnableSearch: model.search, Personalization: settings.Personalization, ReasoningEffort: model.effort,
		IsSaved: settings.ThreadLifecycle.IsSaved(),
	}
	runner, terr := p.startTurn(model.id, settings.ThreadLifecycle.DeletesThreads(), func(ctx context.Context) (kagi.CreatedTurn, error) {
		return p.kagi.CreateTurn(ctx, content, model.id, opts)
	})
	if terr != nil {
		writeError(w, terr)
		return
	}
	// The thread exists before any body byte is written: tell the client which one it is.
	w.Header().Set("X-Kagi-Thread", runner.conversationUUID)
	rs := translate.NewResponseStream(newResponseID(), time.Now().Unix(), model.response)
	p.respondResponses(w, r, runner, rs, req.Stream, req.Tools)
}

// responsesModel is a requested model resolved for one turn.
type responsesModel struct {
	response      string // model echoed back to the client
	id            string // Kagi model id
	search        bool
	effort        string
	contextWindow int
}

// resolveResponsesModel applies the same model, suffix and effort rules as
// /v1/chat/completions.
func (p *Proxy) resolveResponsesModel(r *http.Request, requested, requestEffort string) (responsesModel, *translate.TurnError) {
	catalog, staleReason := p.catalog.get(context.WithoutCancel(r.Context()))
	var m responsesModel
	var spec translate.ModelSpec
	if requested == "" {
		if catalog == nil || catalog.DefaultID == "" {
			return m, translate.NewTurnError(503, "api_error", "kagi_catalog_unavailable", "no model given and the Kagi model catalog is unavailable")
		}
		m.response = catalog.DefaultID
		spec = translate.ModelSpec{Model: catalog.DefaultID, EnableSearch: p.settings.SearchDefaultOnline}
	} else {
		m.response = requested
		var terr *translate.TurnError
		if spec, terr = translate.ParseModel(requested, p.settings.SearchDefaultOnline); terr != nil {
			return m, terr
		}
	}
	m.id, m.search = spec.Model, spec.EnableSearch
	// Per-model effort lists are only trusted from a catalog that is not being served stale.
	var trusted *kagi.Catalog
	if staleReason == "" {
		trusted = catalog
	}
	effort, terr := p.resolveEffort(trusted, m.id, requestEffort, spec.ReasoningEffort)
	if terr != nil {
		return m, terr
	}
	m.effort = effort
	if catalog != nil {
		if info, ok := catalog.Get(m.id); ok {
			m.contextWindow = info.ContextWindow
		}
	}
	return m, nil
}

// respondResponses writes the turn as Responses output. Without tools, text streams as
// it arrives. With tools, the answer is buffered (a TOOL_CALL marker is only
// recognizable in the complete text; streaming clients get keepalives meanwhile) and
// then emitted as function_call items or as a plain message.
func (p *Proxy) respondResponses(w http.ResponseWriter, r *http.Request, runner *turnRunner, rs *translate.ResponseStream, stream bool, tools []translate.ToolSpec) {
	defer runner.consumerGone()
	flusher, _ := w.(http.Flusher)
	write := func(frame string) bool {
		if _, err := io.WriteString(w, frame); err != nil {
			return false // client gone
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	emitText := func(text string) bool {
		if text == "" {
			return true
		}
		if !rs.TextOpen() && !write(rs.OpenText()) {
			return false
		}
		return write(rs.TextDelta(text))
	}
	buffer := len(tools) > 0
	if stream {
		h := w.Header()
		h.Set("Content-Type", "text/event-stream; charset=utf-8")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		if !write(rs.Start()) {
			return
		}
	}

	assembler := translate.NewAssembler()
	var parts []string
	var interrupted *translate.TurnError
	lastEmit := time.Now()
	for {
		var tick <-chan time.Time
		var timer *time.Timer
		if stream {
			timer = time.NewTimer(time.Until(lastEmit.Add(p.settings.Keepalive)))
			tick = timer.C
		}
		it, err := runner.queue.pop(r.Context(), tick)
		if timer != nil {
			timer.Stop()
		}
		if errors.Is(err, errQueueTimeout) {
			if !write(translate.ResponsesKeepalive) {
				return
			}
			lastEmit = time.Now()
			continue
		}
		if err != nil {
			return // client disconnected: the deferred consumerGone arms the bounded drain
		}
		if end, ok := it.(streamEnd); ok {
			interrupted = end.err
			break
		}
		text := assembler.Feed(it.(events.Event))
		parts = append(parts, text)
		if stream && !buffer && text != "" {
			if !emitText(text) {
				return
			}
			lastEmit = time.Now()
		}
		if assembler.Done() {
			break
		}
	}
	tail, terr := assembler.Finish(interrupted)
	parts = append(parts, tail)
	if terr != nil {
		logx.Warn("turn ended with error: %s (%s)", terr.Code(), terr.Message())
		if stream {
			write(rs.Failed(terr))
		} else {
			writeError(w, terr)
		}
		return
	}

	answer := strings.Join(parts, "")
	var calls []translate.ToolCall
	if buffer {
		res := translate.ParseToolCalls(answer, tools)
		if res.Found && res.Err != nil {
			logx.Warn("tool-call marker present but unusable: %v; returning plain content", res.Err)
		}
		if res.Found && res.Err == nil {
			answer, calls = res.Preamble, res.Calls
		}
	}
	if !stream {
		if answer != "" {
			rs.OpenText()
			rs.TextDelta(answer)
			rs.CloseText()
		}
		for _, c := range calls {
			rs.FunctionCall(c)
		}
		writeJSON(w, 200, rs.Body(assembler.Usage))
		return
	}
	pending := tail // without tools, everything before the tail is already streamed
	if buffer {
		pending = answer
	}
	if !emitText(pending) {
		return
	}
	if rs.TextOpen() && !write(rs.CloseText()) {
		return
	}
	for _, c := range calls {
		if !write(rs.FunctionCall(c)) {
			return
		}
	}
	write(rs.Completed(assembler.Usage))
}

func newResponseID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return "resp_" + hex.EncodeToString(b[:])
}
