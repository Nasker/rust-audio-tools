use rinsynth_dsp::voice::VoiceManager;
use rinsynth_model::event::{NoteEvent, SynthEventReceiver};

fn main() {
    let mut vm = VoiceManager::monophonic(44100);
    vm.set_osc_level(1, 1.0);
    vm.set_osc_level(2, 0.0);
    vm.set_osc_level(3, 0.0);
    vm.set_osc_pan(1, -1.0);
    vm.receive_event(NoteEvent::note_on(69, 1.0));

    for i in 0..2000 {
        let s = vm.next_sample_stereo();
        if s.left.is_nan() || s.right.is_nan() {
            println!("NaN at sample {i}: {s:?}");
            break;
        }
        if i < 3 || i == 999 {
            println!("sample {i}: {s:?}");
        }
    }
    println!("done");
}
