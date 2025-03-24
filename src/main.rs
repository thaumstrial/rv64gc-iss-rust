use std::fs::File;
use std::io;
use std::io::Read;
use crate::cpu::CPU;

mod cpu;
mod dram;
mod exception;
mod bus;
mod csr;

fn main() {
    if let Ok(file) = File::open("D:/RustoverProjects/rust-risc-v/src/foo.bin") {
        let mut code = Vec::new();
        let mut file = file;
        file.read_to_end(&mut code).unwrap();
        let mut cpu = CPU::new(code);
        cpu.run();
    } else {
        eprintln!("Failed to open foo.bin");
    }
}
