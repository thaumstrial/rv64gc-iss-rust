use crate::exception::{Exception, Trap};

pub const FFLAGS: u16 = 0x001;  // Floating-point accrued exceptions
pub const FRM: u16 = 0x002;     // Floating-point dynamic rounding mode
pub const FCSR: u16 = 0x003;    // Floating-point Control and Status Register = fflags + frm

pub const UCYCLE: u16 = 0xc00;
pub const UINSTRET: u16 = 0xc02;
pub const MCYCLEH: u16 = 0xc80;
pub const MINSTRETH: u16 = 0xc82;

const SSTATUS_SIE_MASK: u64 = 0x2; // sstatus[1]
const SSTATUS_SPIE_MASK: u64 = 0x20; // sstatus[5]
const SSTATUS_UBE_MASK: u64 = 0x40; // sstatus[6]
const SSTATUS_SPP_MASK: u64 = 0x100; // sstatus[8]
const SSTATUS_FS_MASK: u64 = 0x6000; // sstatus[14:13]
const SSTATUS_XS_MASK: u64 = 0x18000; // sstatus[16:15]
const SSTATUS_SUM_MASK: u64 = 0x40000; // sstatus[18]
const SSTATUS_MXR_MASK: u64 = 0x80000; // sstatus[19]
const SSTATUS_UXL_MASK: u64 = 0x3_00000000; // sstatus[33:32]
const SSTATUS_SD_MASK: u64 = 0x80000000_00000000; // sstatus[63]
const SSTATUS_MASK: u64 = SSTATUS_SIE_MASK
    | SSTATUS_SPIE_MASK
    | SSTATUS_UBE_MASK
    | SSTATUS_SPP_MASK
    | SSTATUS_FS_MASK
    | SSTATUS_XS_MASK
    | SSTATUS_SUM_MASK
    | SSTATUS_MXR_MASK
    | SSTATUS_UXL_MASK
    | SSTATUS_SD_MASK;

const FS_MASK: u64 = 0b11 << 13;
const MSTATUS_WRITE_MASK: u64 =0x0000_0000_FFFF_FFFF | FS_MASK;

pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SCOUNTEREN: u16 = 0x106;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180;

pub const MSTATUS: u16 = 0x300;
pub const MISA: u16 = 0x301;
pub const MVENDORID: u16 = 0xf11; // 0, non-commercial-implementation
pub const MARCHID: u16 = 0xf12; // 0, non-commercial-implementation
pub const MIMPID: u16 = 0xf13; // 0, non-commercial-implementation
pub const MHARTID: u16 = 0xf14; // 0, single-core implementation
pub const MEDELEG: u16 = 0x302;
pub const MIDELEG: u16 = 0x303;
pub const MIE: u16 = 0x304;
pub const MTVEC: u16 = 0x305;
pub const MCOUNTEREN: u16 = 0x306;
pub const MSCRATCH: u16 = 0x340;
pub const MEPC: u16 = 0x341;
pub const MCAUSE: u16 = 0x342;
pub const MTVAL: u16 = 0x343;
pub const MIP: u16 = 0x344;

pub const MCYCLE: u16 = 0xb00;
pub const MINSTRET: u16 = 0xb02;

pub const CSR_COUNT: usize = 4096;

fn check_permission(addr: u16, level: u8, write: bool) -> Result<(), Trap> {
    let csr_priv = ((addr >> 8) & 0x3) as u8;
    if (addr >> 10) == 0b11 && write {
        return Err(Trap::Exception(Exception::IllegalInstruction)); // read-only CSR
    }
    // user: 0, supervisor: 1, machine: 3
    if level < csr_priv {
        return Err(Trap::Exception(Exception::IllegalInstruction)); // insufficient privilege
    }
    Ok(())
}

pub struct CSRs(pub Box<[u64]>);
impl CSRs {
    pub fn new() -> Self {
        let mut csr = CSRs(vec![0; CSR_COUNT].into_boxed_slice());
        csr.0[MISA as usize] =
            // Extensions
            // IMAFDC
            // user mode, supervisor mode
            1 << ('I' as u8 - 'A' as u8)
            | 1 << ('M' as u8 - 'A' as u8)
            | 1 << ('A' as u8 - 'A' as u8)
            | 1 << ('F' as u8 - 'A' as u8)
            | 1 << ('D' as u8 - 'A' as u8)
            | 1 << ('S' as u8 - 'A' as u8)
            | 1 << ('U' as u8 - 'A' as u8)
            // MXL
            // 64
            | (2 << 62);
        csr
    }

    pub fn read(&self, addr: u16, level: u8) -> Result<u64, Trap> {
        check_permission(addr, level, false)?;
        let val = match addr {
            FFLAGS | FRM | FCSR => {
                if ((self.0[MSTATUS as usize] >> 13) & 0b11) == 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction)); // FS = Off
                }
                match addr {
                    FFLAGS => self.0[FFLAGS as usize],
                    FRM => self.0[FRM as usize],
                    FCSR => self.0[FFLAGS as usize] | (self.0[FRM as usize] << 5),
                    _ => unreachable!(),
                }
            }

            SSTATUS => self.0[SSTATUS_MASK as usize] & SSTATUS_MASK,
            SIE => self.0[MIE as usize] & self.0[MIDELEG as usize],
            SIP => self.0[MIP as usize] & self.0[MIDELEG as usize],
            SATP => self.0[SATP as usize],

            MSTATUS => {
                let val = self.0[MSTATUS as usize];
                // State Dirty
                if (val >> 13) & 0b11 == 0b11 || (val >> 15) & 0b11 == 0b11 {
                    val | (1 << 63)
                } else {
                    val
                }
            },
            // because this is a single-core implementation,
            // we can assume that MINSTRET is the same as MCYCLE
            MCYCLE | MINSTRET => self.0[MCYCLE as usize],

            _ => self.0[addr as usize],
        };

        Ok(val)
    }
    pub fn write(&mut self, addr: u16, value: u64, level: u8) -> Result<(), Trap> {
        check_permission(addr, level, true)?;

        match addr {
            MISA => {
                // read-only
            },
            MSTATUS => {
                let old = self.0[MSTATUS as usize];
                let changed = old ^ value;

                if changed & (1 << 17 | 1 << 18 | 1 << 19) != 0 {
                    // tlb_flush_all()
                }

                let fs = (value >> 13) & 0b11;
                if fs > 0b11 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }

                self.0[MSTATUS as usize] = (old & !MSTATUS_WRITE_MASK) | (value & MSTATUS_WRITE_MASK);
            }

            STVEC => self.0[STVEC as usize] = value & !0b11,
            SEPC => self.0[SEPC as usize] = value & !0b1,
            SCAUSE => self.0[SCAUSE as usize] = value,
            SSCRATCH => self.0[SSCRATCH as usize] = value,
            STVAL => self.0[STVAL as usize] = value,
            SIP => {
                let mask = self.0[MIDELEG as usize];
                self.0[MIP as usize] = (self.0[MIP as usize] & !mask) | (value & mask);
            }
            SIE => {
                let mask = self.0[MIDELEG as usize];
                self.0[MIE as usize] = (self.0[MIE as usize] & !mask) | (value & mask);
            }
            MEPC => self.0[MEPC as usize] = value & !0b1,
            MTVEC => {
                // BASE field aligned on a 4-byte boundary
                let changed = (value >> 4) << 4
                    // 0, Direct
                    // 1, Vectored
                    // >=2, Reserved
                    | (value & 0b11);
                self.0[MTVEC as usize] = changed;
            },
            MIDELEG => {
                // medeleg[11] is read-only zero.
                // medeleg[16] is read-only
                self.0[MIDELEG as usize] = value & !(1 << 11 | 1 << 16);
            },
            MIP => {
                // SEIP, STIP, and SSIP are writable
                let mask = (1 << 9) | (1 << 5) | (1 << 1);
                self.0[MIP as usize] = (self.0[MIP as usize] & !mask) | (value & mask);
            }
            _ => self.0[addr as usize] = value,
        }

        Ok(())
    }
    pub fn add_cycle(&mut self) {
        self.0[MCYCLE as usize] += 1;
    }
}