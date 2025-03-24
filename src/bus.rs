use crate::dram::DRAM_SIZE;
use crate::dram::DRAM;
use crate::exception::Exception;

pub const DRAM_BASE: u64 = 0x8000_0000;
const DRAM_END: u64 = DRAM_BASE + DRAM_SIZE;
pub struct Bus {
    pub dram: DRAM,
}
impl Bus {
    pub fn new(data: Vec<u8>) -> Self {
        Bus {
            dram: DRAM::new(data),
        }
    }

    pub fn read(&self, addr: u64, size: u8) -> Result<u64, Exception> {
        match addr {
            DRAM_BASE..= DRAM_END => self.dram.read(addr, size),
            _ => Err(Exception::LoadAccessFault),
        }
    }

    pub fn write(&mut self, addr: u64, value: u64, size: u8) -> Result<(), Exception> {
        match addr {
            DRAM_BASE..=DRAM_END => self.dram.write(addr, value, size),
            _ => Err(Exception::StoreAMOAccessFault),
        }
    }
}