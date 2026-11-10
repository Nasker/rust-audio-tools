# Rust Audio Tools

A reusable Rust audio workspace, extracted from RustInSynth and shared by its
desktop/plugin application and ESP32-S3 firmware.

The initial crates retain their existing names and APIs. This repository is the
single source for their development; applications supply hardware drivers, audio
backends, user interfaces, transport and lifecycle management.

## Crates

| Crate | Responsibility |
| --- | --- |
| `rinsynth-model` | Synth events and types, parameter specifications, preset schema, modulation definitions and factory presets |
| `rinsynth-dsp` | Oscillators, envelopes, filters, distortion, chorus, delay, reverb, arpeggiator, desktop voice engine and static-dispatch embedded engine |

The DSP crate depends on the model crate. Some components are still coupled to
RinSynth's synth model; moving them here does not make every API synth-independent.
General-purpose audio primitives can be separated incrementally as other projects
need them, without changing existing consumers unnecessarily.

## Checkout and local development

The current consumers expect the directory name `audio-tools` next to their own
checkouts. Clone with an explicit destination:

```sh
git clone git@github.com:Nasker/rust-audio-tools.git audio-tools
```

```text
Projects/
├── audio-tools/
│   ├── Cargo.toml
│   └── crates/
│       ├── rinsynth-model/
│       └── rinsynth-dsp/
├── RustInSynth/
└── rinsynth-esp32/
```

Use a stable Rust toolchain for host development. DSP tests use Plotters and its
system font backend, so Linux test builds also need development libraries for
Fontconfig and FreeType.

A sibling application can depend on these crates directly:

```toml
[dependencies]
rinsynth-model = { path = "../audio-tools/crates/rinsynth-model", default-features = false }
rinsynth-dsp = { path = "../audio-tools/crates/rinsynth-dsp" }
```

Local path dependencies immediately use library working-tree changes. For
reproducible releases, pin both crates to the same tested Git revision once it
has been pushed to the remote; local development can use Cargo patches.

## Model features

- Default: no filesystem persistence or desktop directory helpers.
- `fs`: explicit-path preset file operations using `std::fs`.
- `host-fs`: enables `fs` and desktop preset-directory helpers through `dirs`.

The desktop consumer enables `host-fs`. Firmware leaves it disabled.

## Portability and real-time constraints

Both crates currently use `std`; they are not `no_std` libraries. They are used
on desktop and ESP-IDF, but other embedded targets require their own validation.
ESP32 drivers and toolchain configuration are deliberately outside this workspace.

The embedded engine uses static dispatch and inline voice storage. Initialization
allocates effect buffers; sample-rate changes may reallocate delay lines. Do not
assume every control operation is allocation-free simply because sample rendering
is designed for real-time use.

The ESP32 consumer retains a model-crate optimization workaround for Xtensa
serde/f32 code generation. Keep that target-specific setting in the firmware;
the DSP hot path remains optimized.

## Verification

From this repository:

```sh
cargo test --workspace
cargo check --workspace --all-targets --all-features
```

Desktop integration, with the sibling layout above:

```sh
cargo check --manifest-path ../RustInSynth/Cargo.toml --all-targets
cargo check --manifest-path ../RustInSynth/Cargo.toml --lib --features plugin
```

Firmware integration must run from the firmware directory so its toolchain and
Cargo target configuration are active:

```sh
cd ../rinsynth-esp32
cargo +stable test --lib --target x86_64-unknown-linux-gnu
cargo build --release
```

The host-test target above is for x86-64 Linux; use the installed host target on
other platforms. A firmware build additionally requires its ESP-IDF/Xtensa setup.
Host tests and compilation do not prove embedded timing safety: validate render
deadlines, voice headroom, internal heap, stack usage and audible behavior on the
hardware after DSP changes.

## History

The initial crate history was preserved through a Git subtree extraction from
RustInSynth. Earlier commits place the crate directories at repository root; the
workspace setup moves them under `crates/`. Use `git log --follow -- <file>` to
trace individual files across that move.

## License

[The Unlicense](LICENSE), preserved from the original project. Dependencies have
their own licenses.
