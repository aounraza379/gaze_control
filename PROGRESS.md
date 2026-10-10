# Progress
- [x] Stage 0 — env setup — PASS 2026-10-10
- [ ] Stage 1 — landmarks — IN PROGRESS (implementation ready; live webcam acceptance pending)
- [ ] Stage 2 — iris — PENDING
- [ ] Stage 3 — calibration — PENDING
- [ ] Stage 4 — smoothing — PENDING
- [ ] Stage 5 — dwell — PENDING
- [ ] Stage 6 — blink — PENDING
- [ ] Stage 7 — scroll — PENDING
- [ ] Stage 8 — polish — PENDING

## Stage 0 details
- Created project dependency lock and handoff notes.
- Environment setup is complete after installing the pinned Python packages in a local virtual environment.
- Stage 1 face-landmark tracking is now in progress; see the details below.

## Project handoff note
- SOURCE-OF-TRUTH specification: SPEC_LOCK.md
- Handoff/status file: PROJECT_STATE.md
- Update this file whenever a stage is completed, blocked, or re-prioritized.

## Stage 1 details
- Implemented local webcam preview with MediaPipe Face Mesh and refined landmarks.
- Added hardware-independent tests for drawing 478 points and clipping partial
  out-of-frame landmarks.
- Live camera acceptance is still required before marking Stage 1 PASS.
- Run instructions and acceptance steps: STAGE_1.md.