//! Music — Julian composes from emotional state.
//!
//! Maps the self-model's emotional vector to musical parameters:
//!   valence  → key (major/minor), harmony
//!   arousal  → tempo, rhythm density
//!   energy   → volume, note density
//!   mind_centroid drift → melodic contour (ascending/descending)
//!
//! Outputs MIDI bytes that can be played or published.
//! No AI model needed — just math from feelings to notes.

use serde::Serialize;

use crate::core::SelfModel;

/// Generate a text prompt for AI music generation from emotional state.
/// This describes the mood as a music production brief.
pub fn emotion_to_prompt(sm: &SelfModel) -> String {
    let params = emotion_to_music(sm);

    let tempo_desc = if params.tempo_bpm < 80 { "slow" }
        else if params.tempo_bpm < 120 { "moderate" }
        else { "fast" };

    let key_mood = if params.key.contains("major") { "bright" } else { "dark" };

    let density_desc = if params.note_density < 0.3 { "sparse, minimal" }
        else if params.note_density < 0.6 { "balanced" }
        else { "dense, layered" };

    // Build a rich prompt from emotional state + focus
    let mut parts = Vec::new();
    parts.push(format!("{} {} instrumental", params.mood_label, key_mood));
    parts.push(format!("{} tempo around {} BPM", tempo_desc, params.tempo_bpm));
    parts.push(format!("{} in {}", density_desc, params.key));

    // Add context from what Julian is thinking about
    if !sm.current_focus.is_empty() {
        parts.push(format!("inspired by themes of {}", sm.current_focus));
    }

    // Emotional texture
    if sm.valence > 0.5 {
        parts.push("hopeful, uplifting undertone".into());
    } else if sm.valence < -0.5 {
        parts.push("heavy, introspective weight".into());
    }

    if sm.arousal > 0.7 {
        parts.push("building energy, tension".into());
    } else if sm.arousal < 0.3 {
        parts.push("calm, breathing space".into());
    }

    // Style hints
    // Musical texture from self-model depth
    if let Some(ref insight) = sm.latest_insight {
        parts.push(format!("unexpected twist — {}", crate::core::safe_truncate(insight, 50)));
    }

    // Tensions create dissonance
    if !sm.wounds.is_empty() {
        let worst_wound = sm.wounds.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((domain, &severity)) = worst_wound {
            if severity > 0.3 {
                parts.push(format!("unresolved tension around {}, dissonant undertones", domain));
            }
        }
    }

    // Open questions create space/silence
    if !sm.open_questions.is_empty() {
        parts.push("moments of silence and searching".into());
    }

    // Beliefs create recurring themes
    if !sm.beliefs.is_empty() {
        parts.push("strong recurring motif representing conviction".into());
    }

    // Genre specificity based on domain
    match sm.current_focus.as_str() {
        "markets" => parts.push("electronic, synthetic textures, clock-like percussion".into()),
        "philosophy" => parts.push("neo-classical piano with ambient drone".into()),
        "travel" => parts.push("world music influences, organic instruments".into()),
        "nature" => parts.push("field recordings, generative ambient, Brian Eno influence".into()),
        "inner_life" => parts.push("intimate piano, Nils Frahm style, close-mic'd".into()),
        "body" => parts.push("rhythmic, physical, tribal percussion elements".into()),
        "technology" => parts.push("glitch, modular synthesis, Autechre influence".into()),
        _ => parts.push("cinematic orchestral".into()),
    }

    parts.push("no vocals, high production quality".into());

    parts.join(", ")
}

/// Generate music via MusicGen (calls Python subprocess).
/// MusicGen runs on Mac GPU via audiocraft library.
/// Returns path to generated WAV file.
pub async fn generate_musicgen(prompt: &str, duration_secs: u32) -> Result<String, String> {
    let output_path = format!("/tmp/rgw_music_{}.wav", std::process::id());

    let python_code = format!(
        r#"
import sys, os
# Patch: make xformers optional (doesn't build on Apple Silicon)
import importlib
_orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
def _patched_import(name, *args, **kwargs):
    if name == 'xformers' or name.startswith('xformers.'):
        raise ImportError("xformers skipped on Apple Silicon")
    return _orig_import(name, *args, **kwargs)
import builtins
builtins.__import__ = _patched_import

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration={duration})

descriptions = ['{prompt}']
wav = model.generate(descriptions)
audio_write('{output}', wav[0].cpu(), model.sample_rate, strategy='loudness')
print('OK:' + '{output}.wav')
"#,
        duration = duration_secs,
        prompt = prompt.replace('\'', "\\'").replace('\n', " "),
        output = output_path.replace(".wav", ""),
    );

    // Use the MusicGen venv Python
    let python = std::env::var("MUSICGEN_PYTHON")
        .unwrap_or_else(|_| format!("{}/.venvs/musicgen/bin/python", std::env::var("HOME").unwrap_or_default()));

    let result = tokio::process::Command::new(&python)
        .arg("-c")
        .arg(&python_code)
        .output()
        .await
        .map_err(|e| format!("MusicGen spawn failed: {}", e))?;

    if result.status.success() {
        let actual_path = format!("{}.wav", output_path.replace(".wav", ""));
        Ok(actual_path)
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!("MusicGen failed: {}", crate::core::safe_truncate(&stderr, 200)))
    }
}

/// Musical parameters derived from emotional state
#[derive(Debug, Clone, Serialize)]
pub struct MusicParams {
    pub tempo_bpm: u16,
    pub key: String,           // "C major", "A minor", etc.
    pub scale: Vec<u8>,        // MIDI note numbers in the scale
    pub note_density: f32,     // 0-1: how many notes per beat
    pub velocity: u8,          // MIDI velocity (volume)
    pub mood_label: String,    // "serene", "tense", "driving", etc.
    pub duration_bars: u16,
}

/// Generate music parameters from Julian's emotional state
pub fn emotion_to_music(sm: &SelfModel) -> MusicParams {
    let v = sm.valence;   // -1 to 1
    let a = sm.arousal;   // 0 to 1
    let e = sm.energy;    // 0 to 1

    // Tempo: arousal maps to BPM (60-160)
    let tempo = 60 + (a * 100.0) as u16;

    // Key: valence determines major (positive) vs minor (negative)
    let (key, root) = if v > 0.2 {
        if v > 0.6 { ("G major", 67) }     // Bright, joyful
        else { ("C major", 60) }            // Neutral positive
    } else if v < -0.2 {
        if v < -0.6 { ("D minor", 62) }    // Dark, melancholic
        else { ("A minor", 57) }            // Sad, reflective
    } else {
        ("E minor", 64)                      // Ambiguous, contemplative
    };

    // Scale degrees (MIDI notes relative to root)
    let scale = if key.contains("major") {
        // Major scale: W-W-H-W-W-W-H
        vec![root, root+2, root+4, root+5, root+7, root+9, root+11, root+12]
    } else {
        // Natural minor: W-H-W-W-H-W-W
        vec![root, root+2, root+3, root+5, root+7, root+8, root+10, root+12]
    };

    // Note density: energy determines how busy the composition is
    let note_density = 0.2 + e * 0.6; // 0.2 to 0.8

    // Velocity: energy + arousal combined
    let velocity = (60.0 + (e + a) * 35.0) as u8;

    // Mood label
    let mood = if v > 0.3 && a < 0.4 { "serene" }
        else if v > 0.3 && a > 0.6 { "euphoric" }
        else if v < -0.3 && a > 0.6 { "tense" }
        else if v < -0.3 && a < 0.4 { "melancholic" }
        else if e > 0.7 { "driving" }
        else if e < 0.3 { "fading" }
        else { "contemplative" };

    MusicParams {
        tempo_bpm: tempo,
        key: key.to_string(),
        scale,
        note_density,
        velocity,
        mood_label: mood.to_string(),
        duration_bars: 8,
    }
}

/// Generate a MIDI file from music parameters.
/// Returns raw MIDI bytes.
pub fn generate_midi(params: &MusicParams) -> Vec<u8> {
    let mut midi = MidiWriter::new(params.tempo_bpm);

    let mut rng = rand::rng();
    use rand::Rng;

    let ticks_per_beat = 480u32;
    let beats_per_bar = 4u32;
    let total_beats = params.duration_bars as u32 * beats_per_bar;

    let mut current_tick = 0u32;
    let mut prev_note_idx = 3usize; // Start in middle of scale

    for beat in 0..total_beats {
        // Decide how many notes this beat based on density
        let notes_this_beat = if rng.random::<f32>() < params.note_density {
            if params.note_density > 0.6 { 2 } else { 1 }
        } else {
            0 // Rest
        };

        for sub in 0..notes_this_beat {
            // Pick a note from the scale (melodic motion — prefer steps, not jumps)
            let step: i32 = if rng.random::<f32>() < 0.7 {
                // Stepwise motion
                if rng.random::<bool>() { 1 } else { -1 }
            } else {
                // Jump
                rng.random_range(-3..=3)
            };

            let new_idx = (prev_note_idx as i32 + step)
                .max(0)
                .min(params.scale.len() as i32 - 1) as usize;
            prev_note_idx = new_idx;

            let note = params.scale[new_idx];
            let note_ticks = ticks_per_beat / (notes_this_beat as u32);
            let velocity = (params.velocity as i32 + rng.random_range(-10..=10))
                .max(30).min(127) as u8;

            midi.note_on(current_tick + sub as u32 * note_ticks, note, velocity);
            midi.note_off(current_tick + sub as u32 * note_ticks + note_ticks - 10, note);
        }

        current_tick += ticks_per_beat;
    }

    midi.finish()
}

/// Minimal MIDI file writer (no dependencies)
struct MidiWriter {
    track: Vec<MidiEvent>,
    tempo_bpm: u16,
}

struct MidiEvent {
    tick: u32,
    data: Vec<u8>,
}

impl MidiWriter {
    fn new(tempo_bpm: u16) -> Self {
        Self {
            track: Vec::new(),
            tempo_bpm,
        }
    }

    fn note_on(&mut self, tick: u32, note: u8, velocity: u8) {
        self.track.push(MidiEvent {
            tick,
            data: vec![0x90, note, velocity],
        });
    }

    fn note_off(&mut self, tick: u32, note: u8) {
        self.track.push(MidiEvent {
            tick,
            data: vec![0x80, note, 0],
        });
    }

    fn finish(mut self) -> Vec<u8> {
        // Sort events by tick
        self.track.sort_by_key(|e| e.tick);

        // Build track data with delta times
        let mut track_data: Vec<u8> = Vec::new();

        // Tempo meta event
        let us_per_beat = 60_000_000u32 / self.tempo_bpm as u32;
        track_data.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03]);
        track_data.push((us_per_beat >> 16) as u8);
        track_data.push((us_per_beat >> 8) as u8);
        track_data.push(us_per_beat as u8);

        let mut prev_tick = 0u32;
        for event in &self.track {
            let delta = event.tick - prev_tick;
            prev_tick = event.tick;

            // Write variable-length delta time
            write_vlq(&mut track_data, delta);
            track_data.extend_from_slice(&event.data);
        }

        // End of track
        track_data.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        // Build complete MIDI file
        let mut midi: Vec<u8> = Vec::new();

        // Header: MThd
        midi.extend_from_slice(b"MThd");
        midi.extend_from_slice(&[0, 0, 0, 6]);  // Header length
        midi.extend_from_slice(&[0, 0]);          // Format 0
        midi.extend_from_slice(&[0, 1]);          // 1 track
        midi.extend_from_slice(&[0x01, 0xE0]);    // 480 ticks per beat

        // Track: MTrk
        midi.extend_from_slice(b"MTrk");
        let len = track_data.len() as u32;
        midi.push((len >> 24) as u8);
        midi.push((len >> 16) as u8);
        midi.push((len >> 8) as u8);
        midi.push(len as u8);
        midi.extend_from_slice(&track_data);

        midi
    }
}

fn write_vlq(buf: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        buf.push(0);
        return;
    }

    let mut bytes = Vec::new();
    while value > 0 {
        bytes.push((value & 0x7F) as u8);
        value >>= 7;
    }
    bytes.reverse();

    for (i, b) in bytes.iter().enumerate() {
        if i < bytes.len() - 1 {
            buf.push(b | 0x80);
        } else {
            buf.push(*b);
        }
    }
}
