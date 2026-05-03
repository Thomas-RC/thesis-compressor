use anyhow::Result;
use candle_core::Device;
use thesis_compressor::data::{BatchSampler, ENWIK8_TOTAL, load_enwik8};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enwik8".to_string());

    println!("[+] wczytuję enwik8 z: {path}");
    let splits = load_enwik8(&path)?;
    println!(
        "    train: {:>10} B | val: {:>9} B | test: {:>9} B | total: {} B",
        splits.train.len(),
        splits.val.len(),
        splits.test.len(),
        ENWIK8_TOTAL
    );

    let preview: String = splits.train[..120]
        .iter()
        .map(|&b| {
            if b.is_ascii() && !b.is_ascii_control() {
                b as char
            } else {
                '.'
            }
        })
        .collect();
    println!("    preview train[..120]:\n      {preview:?}");

    let unique = {
        let mut seen = [false; 256];
        for &b in splits.train.iter().take(1_000_000) {
            seen[b as usize] = true;
        }
        seen.iter().filter(|&&x| x).count()
    };
    println!("    distinct byte values w pierwszym 1MB train: {unique}/256");

    let device = Device::Cpu;
    let seq_len = 256;
    let batch_size = 4;
    let mut sampler = BatchSampler::new(&splits.train, seq_len, batch_size, 42);

    println!("[+] losuję paczkę: batch={batch_size}, seq_len={seq_len}");
    let (input, target) = sampler.sample(&device)?;
    println!("    input  shape={:?} dtype={:?}", input.shape(), input.dtype());
    println!("    target shape={:?} dtype={:?}", target.shape(), target.dtype());

    let input_vec: Vec<Vec<u32>> = input.to_vec2()?;
    let target_vec: Vec<Vec<u32>> = target.to_vec2()?;
    let mut mismatches = 0usize;
    for b in 0..batch_size {
        for t in 0..seq_len - 1 {
            if input_vec[b][t + 1] != target_vec[b][t] {
                mismatches += 1;
            }
        }
    }
    println!(
        "[+] sanity-check (target == input shifted by +1): {}",
        if mismatches == 0 {
            "OK".to_string()
        } else {
            format!("FAIL ({mismatches} niezgodności)")
        }
    );

    let row0_decoded: String = input_vec[0]
        .iter()
        .take(60)
        .map(|&u| {
            let b = u as u8;
            if b.is_ascii() && !b.is_ascii_control() {
                b as char
            } else {
                '.'
            }
        })
        .collect();
    println!("    input[0, ..60] jako ASCII: {row0_decoded:?}");

    println!("[+] smoke test zakończony pomyślnie");
    Ok(())
}
