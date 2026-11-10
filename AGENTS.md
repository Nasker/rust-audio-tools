# audio-tools

Independent Rust audio workspace shared by sibling RustInSynth (desktop/plugin)
and rinsynth-esp32 (ESP32-S3 firmware) repositories.

## Boundaries

- `crates/rinsynth-model`: synth events, parameter specification, presets,
  modulation types and factory bank. No filesystem support by default.
  `fs` adds explicit-path persistence; `host-fs` adds desktop directory helpers.
- `crates/rinsynth-dsp`: oscillators, envelopes, filters, effects, arpeggiator,
  desktop voice engine and static-dispatch embedded engine. Depends on the model.
- Both crates currently use std; portability does not imply no_std support.
- Keep hardware drivers, UI, transport queues and application lifecycle in consumers.
- Keep crate names/APIs stable during extraction. General-purpose primitives can
  be separated from synth-specific model dependencies incrementally, not wholesale.

## Verification

From this repository:

```sh
cargo test --workspace
cargo check --workspace --all-targets --all-features
```

Consumer compatibility (sibling layout):

```sh
cargo check --manifest-path ../RustInSynth/Cargo.toml --all-targets
cargo check --manifest-path ../RustInSynth/Cargo.toml --lib --features plugin
```

From `../rinsynth-esp32` (its toolchain/config must be active):

```sh
cargo +stable test --lib --target x86_64-unknown-linux-gnu
cargo build --release
```

Host tests do not prove embedded timing safety. After DSP changes check hardware
render deadlines, voice headroom, heap and stack usage before declaring a release.
No allocations/locks in the per-sample hot path; retain inline voice storage,
static dispatch and IRAM placement. The ESP32 consumer retains the model crate's
opt-level=0 workaround for Xtensa serde-f32 code generation; DSP remains optimized.

## Dependency and history policy

Consumers currently use sibling path dependencies into this workspace. Once a
remote exists, reproducible releases should pin both crates to the same Git
revision; local development can use Cargo patches. No remote is configured yet.
Do not publish or guess a remote URL. Do not maintain a second editable copy in
an application repository.

History was extracted from RustInSynth's `crates/` subtree at source commit
`d30718b` (split commit `d624512eac8d046b87ebb18fb3c682c6653c2600`). Historical
subtree commits have the crate directories at repository root; the extraction
setup moves them under `crates/`. Use `git log --follow` to trace file history.
The source repository's Unlicense is preserved in LICENSE.
