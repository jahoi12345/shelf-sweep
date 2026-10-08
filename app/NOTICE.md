# Attribution and reuse record (app/)

Portions of `app/` are adapted from **Insurance Claim Live Agent Team**
(`voice_ai_agents/insurance_claim_live_agent_team` in https://github.com/Shubhamsaboo/awesome-llm-apps),
pinned at commit `4bf51ab704fb2c5b3803cd5191b30d7dcdb51dc2`, licensed **Apache-2.0** (`app/LICENSE-upstream-Apache-2.0`).
The unmodified snapshot with SHA-256 hashes is in `third_party/insurance_claim_live_agent_team/`. Adapted files carry a
header comment stating they are modified. This notice is the summary of what was kept, changed and discarded.

| Upstream file / function | Status | What happened |
| --- | --- | --- |
| `live_demo/server.py` app skeleton, `local_access` middleware, owner cookie, `owned_session`, `cleanup_sessions`, `discard_session`, startup/shutdown cleanup | **Kept** (adapted names) | Same local-only/origin/size checks, session TTL and ownership; `discard_session` also awaits the engine and can purge evidence |
| `live_voice` WebSocket handler: `send` lock, `track`, `finalize`, `client_to_gemini` loop (input windows/rate limits, size limits, base64/PCM checks), `gemini_to_client` loop (transcript chunks, `interrupted`, tool-call launch/cancellation, `turn_complete`), `launch_tool`/`execute_tool` with `FunctionResponseScheduling` | **Kept, adapted** | Rewritten as `client_loop`/`model_loop` with the same logic; audio forwarding only (no avatar video); optional Live session so the app degrades without a credential |
| `IntakeSession`, `append_turn` limits, turn ids | **Kept, adapted** | `SweepSession` owns an incremental engine instead of an FNOL claim |
| `_pin_evidence_photo` / `FrameObservation` (freeze exact frame, server-side independent caption) | **Generalised** | Frames are persisted unchanged by the engine at ingestion; detect/read stages are the independent model checks; no insurer caption model |
| `live_demo/live_tools.py`: `build_live_config` (barge-in `START_OF_ACTIVITY_INTERRUPTS`, transcription), `tool_declarations` shape, `scheduling_for`, camera-state notices | **Kept, adapted** | New prompt, new tools (`get_inventory_status`, `record_correction`, `confirm_market`, `end_sweep`); capture notices reuse the camera-state notice pattern |
| `live_demo/app.js`: session/connect flow with generation counter, `connectLive`, mic capture/resample/PCM16, `playPcm24`/`stopPlayback`, `startCamera`/`stopCamera`, transcript merge, tool-activity merge, teardown | **Kept** (near verbatim) | Frame sending split into a 512 px voice-context frame and a high-resolution evidence frame; `startReplay` added on the same frame path; notebook/policy/avatar rendering replaced by inventory rendering |
| `live_demo/index.html`, `styles.css` | **Kept layout/styles, content replaced** | Same three-column desk layout and card/pill/control/transcript styles; inventory/review styles appended |
| `tests/client-regressions.cjs` | **Ported** | `tests/client-regressions-app.cjs`: connection, interruption, reset, media release, camera mode, plus new replay/evidence/unknown-value tests |
| `tests/test_live_turns.py` | **Ported** | Interrupted-payload ordering test reproduced in `tests/test_live_app.py` |
| `tests/test_regressions.py` (workflow-in-flight/stale, cleanup, access) | **Ported in spirit** | Stale-merge guard, cleanup/purge, replay path traversal, packet-ZIP gating |
| `agent.py` ADK graph, `policies.py`, `policy_directory.py`, `schemas.py`, `examples.py` | **Discarded** | Insurance claim rules/policies; our separable stage functions replace them. ADK not adopted (no benefit demonstrated) |
| Avatar (`avatar.js`, avatar config, `live_media_message` video path), sketch generation, mock policy lookup, FNOL packet markdown | **Discarded** | Out of scope; remain only in the unmodified snapshot |

New code (not from upstream): `app/engine.py` (incremental ingestion, selection, nudges, revision-guarded merge, finalization),
`app/replay.py`, `app/smoke.py`, `pipeline/corrections.py`, the packet/report extensions and all stage/measurement/pricing logic.
