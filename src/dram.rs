use crate::bus::DRAM_BASE;
use crate::exception::Exception;
use crate::cpu::{BYTE, HALF_WORD, WORD, DOUBLE_WORD};

pub const DRAM_SIZE: u64 = 1024 * 1024 * 128; // 128MiB
pub struct DRAM(Box<[u8]>);
impl DRAM {
    pub fn new(binary: Vec<u8>) -> Self {
        let mut data = vec![0; DRAM_SIZE as usize];
        data[..binary.len()].copy_from_slice(&binary);
        DRAM(data.into_boxed_slice())
    }

    pub fn read(&self, addr: u64, size: u8) -> Result<u64, Exception> {
        match size {
            BYTE => Ok(self.read8(addr)),
            HALF_WORD => Ok(self.read16(addr)),
            WORD => Ok(self.read32(addr)),
            DOUBLE_WORD => Ok(self.read64(addr)),
            _ => Err(Exception::LoadAccessFault),
        }
    }

    pub fn write(&mut self, addr: u64, value: u64, size: u8) -> Result<(), Exception> {
        match size {
            BYTE => Ok(self.write8(addr, value)),
            HALF_WORD => Ok(self.write16(addr, value)),
            WORD => Ok(self.write32(addr, value)),
            DOUBLE_WORD => Ok(self.write64(addr, value)),
            _ => Err(Exception::StoreAMOAccessFault),
        }
    }

    fn write8(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.0[index] = val as u8
    }

    fn write16(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.0[index] = (val & 0xff) as u8;
        self.0[index + 1] = ((val >> 8) & 0xff) as u8;
    }

    fn write32(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.0[index] = (val & 0xff) as u8;
        self.0[index + 1] = ((val >> 8) & 0xff) as u8;
        self.0[index + 2] = ((val >> 16) & 0xff) as u8;
        self.0[index + 3] = ((val >> 24) & 0xff) as u8;
    }

    fn write64(&mut self, addr: u64, val: u64) {
        let index = (addr - DRAM_BASE) as usize;
        self.0[index] = (val & 0xff) as u8;
        self.0[index + 1] = ((val >> 8) & 0xff) as u8;
        self.0[index + 2] = ((val >> 16) & 0xff) as u8;
        self.0[index + 3] = ((val >> 24) & 0xff) as u8;
        self.0[index + 4] = ((val >> 32) & 0xff) as u8;
        self.0[index + 5] = ((val >> 40) & 0xff) as u8;
        self.0[index + 6] = ((val >> 48) & 0xff) as u8;
        self.0[index + 7] = ((val >> 56) & 0xff) as u8;
    }

    fn read8(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        self.0[index] as u64
    }

    fn read16(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.0[index] as u64) | ((self.0[index + 1] as u64) << 8);
    }

    fn read32(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.0[index] as u64)
            | ((self.0[index + 1] as u64) << 8)
            | ((self.0[index + 2] as u64) << 16)
            | ((self.0[index + 3] as u64) << 24);
    }

    fn read64(&self, addr: u64) -> u64 {
        let index = (addr - DRAM_BASE) as usize;
        return (self.0[index] as u64)
            | ((self.0[index + 1] as u64) << 8)
            | ((self.0[index + 2] as u64) << 16)
            | ((self.0[index + 3] as u64) << 24)
            | ((self.0[index + 4] as u64) << 32)
            | ((self.0[index + 5] as u64) << 40)
            | ((self.0[index + 6] as u64) << 48)
            | ((self.0[index + 7] as u64) << 56);
    }

}