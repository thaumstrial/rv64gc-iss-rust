use crate::bus::{Bus, DRAM_BASE};
use crate::csr::CSRs;
use crate::exception::Exception;

pub const REGS_COUNT: usize = 32;
pub const BYTE: u8 = 8;
pub const HALF_WORD: u8 = 16;
pub const WORD: u8 = 32;
pub const DOUBLE_WORD: u8 = 64;
pub fn sign_extend(val: u64, bits: u8) -> u64 {
    let shift = 64 - bits;
    ((val << shift) as i64 >> shift) as u64
}
pub fn decode32_r(inst: u32) -> (u64, u64, u64, u8, u8) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let rs2 = ((inst >> 20) & 0x1f) as u64;
    let funct7 = ((inst >> 25) & 0x7f) as u8;
    (rd, rs1, rs2, funct3, funct7)
}

// sign-extend in execution
pub fn decode32_i(inst: u32) -> (u64, u64, u8, u16) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let imm = ((inst >> 20) & 0xfff) as u16;
    (rd, rs1, funct3, imm)
}
// sign-extend in execution
pub fn decode32_s(inst: u32) -> (u64, u64, u8, u16) {
    let imm = (((inst >> 25) << 5) | ((inst >> 7) & 0x1f)) as u16;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let rs2 = ((inst >> 20) & 0x1f) as u64;
    (rs1, rs2, funct3, imm)
}

pub fn decode32_b(inst: u32) -> (u64, u64, u8, u64) {
    let imm = {
        let imm12 = ((inst >> 31) & 1) << 12;
        let imm11 = ((inst >> 7) & 1) << 11;
        let imm10_5 = ((inst >> 25) & 0x3f) << 5;
        let imm4_1 = ((inst >> 8) & 0xf) << 1;
        (imm12 | imm11 | imm10_5 | imm4_1) as u16
    };
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let rs2 = ((inst >> 20) & 0x1f) as u64;
    (rs1, rs2, funct3, sign_extend(imm as u64, 13))
}

pub fn decode32_u(inst: u32) -> (u64, u64) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let imm = (inst & 0xfffff000) as u64;
    (rd, imm)
}

pub fn decode32_j(inst: u32) -> (u64, u64) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let imm = {
        let imm20 = ((inst >> 31) & 1) << 20;
        let imm19_12 = ((inst >> 12) & 0xff) << 12;
        let imm11 = ((inst >> 20) & 1) << 11;
        let imm10_1 = ((inst >> 21) & 0x3ff) << 1;
        (imm20 | imm19_12 | imm11 | imm10_1)
    };
    (rd, sign_extend(imm as u64, 21))
}

pub fn decode16_cr(inst: u16) -> (u64, u64, u8) {
    let rd_rs1 = ((inst >> 7) & 0x1f) as u64;
    let rs2 = ((inst >> 2) & 0x1f) as u64;
    let funct4 = ((inst >> 12) & 0xf) as u8;
    (rs2, rd_rs1, funct4)
}
pub fn decode16_ci(inst: u16) -> (u64, u16, u8) {
    let func3 = ((inst >> 13) & 0x7) as u8;
    let rd_rs1 = ((inst >> 7) & 0x1f) as u64;
    let imm = (((inst >> 2) & 0x1f) | ((inst >> 12) << 5));
    (rd_rs1, imm, func3)
}
pub fn decode16_css(inst: u16) -> (u64, u16, u8) {
    let func3 = ((inst >> 13) & 0x7) as u8;
    let rs2 = ((inst >> 2) & 0x1f) as u64;
    let imm = (((inst >> 7) & 0x3f) | ((inst >> 12) << 6)) as u16;
    (rs2, imm, func3)
}
pub fn decode16_ciw(inst: u16) -> (u64, u16, u8) {
    let func3 = ((inst >> 13) & 0x7) as u8;
    let rd = 8 + ((inst >> 2) & 0x7) as u64;
    let imm = (((inst >> 5) & 0x1) << 3)
        | (((inst >> 6) & 0x1) << 2)
        | (((inst >> 2) & 0x3) << 6)
        | (((inst >> 12) & 0x1) << 5)
        | (((inst >> 3) & 0x3) << 4);
    (rd, imm,func3)
}
pub fn decode16_cl(inst: u16) -> (u64, u64, u16, u8) {
    let func3 = ((inst >> 13) & 0x7) as u8;
    let rd = 8 + ((inst >> 2) & 0x7) as u64;
    let rs1 = 8 + ((inst >> 7) & 0x7) as u64;
    let imm = (((inst >> 5) & 0x1) << 3) | (((inst >> 10) & 0x7) << 6) | (((inst >> 6) & 0x1) << 2);
    (rd, rs1, imm as u16, func3)
}
pub fn decode16_cs(inst: u16) -> (u64, u64, u16, u8) {
    let func3 = ((inst >> 13) & 0x7) as u8;
    let rs2 = 8 + ((inst >> 2) & 0x7) as u64;
    let rs1 = 8 + ((inst >> 7) & 0x7) as u64;
    let imm = (((inst >> 5) & 0x1) << 3) | (((inst >> 10) & 0x7) << 6) | (((inst >> 6) & 0x1) << 2);
    (rs1, rs2, imm as u16, func3)
}
pub fn decode16_ca(inst: u16) -> (u64, u64, u8) {
    let rd_rs1 = 8 + ((inst >> 7) & 0x7) as u64;
    let rs2 = 8 + ((inst >> 2) & 0x7) as u64;
    let funct6 = ((inst >> 10) & 0x3f) as u8;
    (rd_rs1, rs2, funct6)
}
pub fn decode16_cb(inst: u16) -> (u64, u64) {
    let rs1 = 8 + ((inst >> 7) & 0x7) as u64;
    let imm = sign_extend(
        (((inst >> 2) & 0x1f) | ((inst >> 12) << 5)) as u64,
        6,
    );
    (rs1, imm)
}

pub fn decode16_cj(inst: u16) -> u64 {
    let imm = sign_extend(
        ((((inst >> 12) & 0x1) << 11)
            | (((inst >> 11) & 0x1) << 4)
            | (((inst >> 9) & 0x3) << 8)
            | (((inst >> 8) & 0x1) << 10)
            | (((inst >> 7) & 0x1) << 6)
            | (((inst >> 6) & 0x1) << 7)
            | (((inst >> 3) & 0x7) << 1)
            | (((inst >> 2) & 0x1) << 5)) as u64,
        12,
    );
    imm
}
pub struct XRegs([u64; REGS_COUNT]);
impl XRegs {
    pub fn new() -> XRegs {
        XRegs([0; REGS_COUNT])
    }

    pub fn read(&self, i: u64) -> u64 {
        self.0[i as usize]
    }

    pub fn write(&mut self, i: u64, v: u64) {
        if i != 0 {
            self.0[i as usize] = v;
        }
    }
}
// RV64IMAFDCZicsr_Zifencei
pub struct CPU {
    pub x_regs: XRegs,
    pub bus: Bus,
    pub pc: u64,
    pub csr: CSRs,
    /// user: 0, supervisor: 1, machine: 3
    pub level: u8,
}
impl CPU {
    pub fn new(data: Vec<u8>) -> Self {
        CPU {
            x_regs: XRegs::new(),
            bus: Bus::new(data),
            pc: DRAM_BASE,
            csr: CSRs::new(),
            level: 0b11,
        }
    }

    pub fn run(&mut self) {
        loop {
            if let Err(e) = self.step() {
                match e {
                    Exception::Breakpoint => {
                        println!("Breakpoint");
                        break;
                    },
                    Exception::EnvironmentCallFromUMode => {
                        println!("EnvironmentCallFromUMode");
                        break;
                    },
                    Exception::EnvironmentCallFromSMode => {
                        println!("EnvironmentCallFromSMode");
                        break;
                    },
                    Exception::EnvironmentCallFromMMode => {
                        println!("EnvironmentCallFromMMode");
                        break;
                    },
                    _ => {
                        println!("Error");
                        break;
                    }
                }
            }
        }
    }

    pub fn step(&mut self) -> Result<(), Exception> {
        // fetch
        let inst16 = self.bus.read(self.pc, HALF_WORD)? as u16;
        if inst16 & 0b11 == 0b11 {
            let inst32 = self.bus.read(self.pc, WORD)? as u32;
            self.execute32(inst32)?;
        } else {
            self.execute16(inst16)?;
        };

        self.csr.add_cycle();
        Ok(())
    }

    pub fn next_inst32(&mut self){
        self.pc = self.pc.wrapping_add(4);
    }
    pub fn next_inst16(&mut self){
        self.pc = self.pc.wrapping_add(2);
    }
    pub fn jump_inst(&mut self, offset: u64) {
        self.pc = self.pc.wrapping_add(offset);
    }

    pub fn execute32(&mut self, inst: u32) -> Result<(), Exception> {
        let opcode = inst & 0x7f;
        match opcode {
            0x37 => { // LUI
                let (rd, imm) = decode32_u(inst);
                self.x_regs.write(rd, imm);
                self.next_inst32();
            },
            0x17 => { // AUIPC
                let (rd, imm) = decode32_u(inst);
                self.x_regs.write(rd, self.pc.wrapping_add(imm));
                self.next_inst32();
            },
            0x6f => { // JAL
                let (rd, offset) = decode32_j(inst);
                self.x_regs.write(rd, self.pc.wrapping_add(4));
                self.jump_inst(offset);
            },
            0x67 => { // JALR
                let (rd, rs1, funct3, imm) = decode32_i(inst);
                if funct3 != 0 {
                    return Err(Exception::IllegalInstruction);
                }
                let target = self.x_regs.read(rs1)
                    .wrapping_add(sign_extend(imm as u64, 12)) & !1;
                self.x_regs.write(rd, self.pc.wrapping_add(4));
                self.pc = target;
            },
            0x63 => { // branch
                let (rs1, rs2, funct3, offset) = decode32_b(inst);
                let val1 = self.x_regs.read(rs1);
                let val2 = self.x_regs.read(rs2);
                let take_branch = match funct3 {
                    0b000 => val1 == val2,                         // BEQ
                    0b001 => val1 != val2,                         // BNE
                    0b100 => (val1 as i64) < (val2 as i64),        // BLT
                    0b101 => (val1 as i64) >= (val2 as i64),       // BGE
                    0b110 => val1 < val2,                          // BLTU
                    0b111 => val1 >= val2,                         // BGEU
                    _ => unreachable!(),
                };

                if take_branch {
                    self.jump_inst(offset);
                } else {
                    self.next_inst32();
                }
            },
            0x03 => { // load
                let (rd, rs1, funct3, imm) = decode32_i(inst);
                let offset = sign_extend(imm as u64, 12);

                let addr = self.x_regs.read(rs1).wrapping_add(offset);
                let val = match funct3 {
                    0b000 => self.bus.read(addr, BYTE)? as i32 as i64 as u64, // LB
                    0b001 => self.bus.read(addr, HALF_WORD)? as i32 as i64 as u64, // LH
                    0b010 => self.bus.read(addr, WORD)? as i32 as i64 as u64, // LW
                    0b011 => self.bus.read(addr, DOUBLE_WORD)?, // LD (RV64I)
                    0b100 => self.bus.read(addr, BYTE)?, // LBU
                    0b101 => self.bus.read(addr, HALF_WORD)?, // LHU
                    0b110 => self.bus.read(addr, WORD)?, // LWU (RV64I)
                    _ => unreachable!(),
                };
                self.x_regs.write(rd, val);
                self.next_inst32();
            },
            0x23 => { // store
                let (rs1, rs2, funct3, imm) = decode32_s(inst);
                let offset = sign_extend(imm as u64, 12);
                let addr = self.x_regs.read(rs1).wrapping_add(offset);
                let val = self.x_regs.read(rs2);
                match funct3 {
                    0b000 => self.bus.write(addr, val, BYTE)?, // SB
                    0b001 => self.bus.write(addr, val, HALF_WORD)?, // SH
                    0b010 => self.bus.write(addr, val, WORD)?, // SW
                    0b011 => self.bus.write(addr, val, DOUBLE_WORD)?, // SD (RV64I)
                    _ => return Err(Exception::IllegalInstruction),
                }
                self.next_inst32();
            },
            0x13 => { // I-type ALU
                let (rd, rs1, funct3, imm_raw) = decode32_i(inst);
                let rs1_val = self.x_regs.read(rs1);
                let imm = sign_extend(imm_raw as u64, 12);
                let result = match funct3 {
                    0b000 => rs1_val.wrapping_add(imm),                      // ADDI
                    0b010 => ((rs1_val as i64) < (imm as i64)) as u64,       // SLTI
                    0b011 => (rs1_val < imm) as u64,                         // SLTIU
                    0b100 => rs1_val ^ imm,                                  // XORI
                    0b110 => rs1_val | imm,                                  // ORI
                    0b111 => rs1_val & imm,                                  // ANDI
                    0b001 => { // SLLI
                        let shamt = (imm_raw & 0x3f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        if funct7 != 0b0000000 {
                            return Err(Exception::IllegalInstruction);
                        }
                        rs1_val << shamt
                    }
                    0b101 => {
                        let shamt = (imm_raw & 0x3f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        match funct7 {
                            0b0000000 => rs1_val >> shamt,                    // SRLI
                            0b0100000 => ((rs1_val as i64) >> shamt) as u64, // SRAI
                            _ => return Err(Exception::IllegalInstruction),
                        }
                    },
                    _ => return Err(Exception::IllegalInstruction),
                };

                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x33 => { // R-Type ALU
                let (rd, rs1, rs2, funct3, funct7) = decode32_r(inst);
                let val1 = self.x_regs.read(rs1);
                let val2 = self.x_regs.read(rs2);
                let result = match (funct3, funct7) {
                    (0b000, 0b0000000) => val1.wrapping_add(val2),               // ADD
                    (0b000, 0b0100000) => val1.wrapping_sub(val2),               // SUB
                    (0b001, 0b0000000) => val1 << (val2 & 0x3f),                  // SLL
                    (0b010, 0b0000000) => ((val1 as i64) < (val2 as i64)) as u64, // SLT
                    (0b011, 0b0000000) => (val1 < val2) as u64,                   // SLTU
                    (0b100, 0b0000000) => val1 ^ val2,                            // XOR
                    (0b101, 0b0000000) => val1 >> (val2 & 0x3f),                  // SRL
                    (0b101, 0b0100000) => ((val1 as i64) >> (val2 & 0x3f)) as u64,// SRA
                    (0b110, 0b0000000) => val1 | val2,                            // OR
                    (0b111, 0b0000000) => val1 & val2,                            // AND
                    _ => return Err(Exception::IllegalInstruction),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x0f => { // FENCE, FENCE.TSO, PAUSE
                // ignored in the single-core implementation
                let (_, _, funct3, _) = decode32_i(inst);
                match funct3 {
                    0b000 => self.next_inst32(),  // FENCE
                    0b001 => self.next_inst32(),  // FENCE.I (RV32/RV64 Fiencei)
                    _ => return Err(Exception::IllegalInstruction),
                }
            },
            0x73 => { // SYSTEM
                let (rd, rs1, funct3, imm) = decode32_i(inst);
                let csr = imm;
                let uimm = rs1;

                match funct3 {
                    0b000 => {
                        if !(rd != 0 && rs1 == 0) {
                            return Err(Exception::IllegalInstruction);
                        }
                        return match imm {
                            0x000 => { // ECALL
                                match self.level {
                                    0b00 => {
                                        Err(Exception::EnvironmentCallFromUMode)
                                    },
                                    0b01 => {
                                        Err(Exception::EnvironmentCallFromSMode)
                                    },
                                    0b11 => {
                                        Err(Exception::EnvironmentCallFromMMode)
                                    },
                                    _ => { unreachable!() },
                                }
                            },
                            0x001 => { // EBREAK
                                Err(Exception::Breakpoint)
                            },
                            _ => Err(Exception::IllegalInstruction),
                        }
                    },
                    // Zicsr Standard Extension
                    0b001 => { // CSRRW
                        let write_val = self.x_regs.read(rs1);
                        if rd != 0 {
                            let old = self.csr.read(csr, self.level)?;
                            self.x_regs.write(rd, old);
                        }
                        self.csr.write(csr, write_val, self.level)?;
                        self.next_inst32();
                    },
                    0b010 => { // CSRRS
                        let old = self.csr.read(csr, self.level)?;
                        if rs1 != 0 {
                            self.csr.write(csr, old | self.x_regs.read(rs1), self.level)?;
                        }
                        self.x_regs.write(rd, old);
                        self.next_inst32();
                    },
                    0b011 => { // CSRRC
                        let old = self.csr.read(csr, self.level)?;
                        if rs1 != 0 {
                            self.csr.write(csr, old & !self.x_regs.read(rs1), self.level)?;
                        }
                        self.x_regs.write(rd, old);
                        self.next_inst32();
                    },
                    0b101 => { // CSRRWI
                        if rd != 0 {
                            let old = self.csr.read(csr, self.level)?;
                            self.x_regs.write(rd, old);
                        }
                        self.csr.write(csr, uimm, self.level)?;
                        self.next_inst32();
                    },
                    0b110 => { // CSRRSI
                        let old = self.csr.read(csr, self.level)?;
                        if uimm != 0 {
                            self.csr.write(csr, old | uimm, self.level)?;
                        }
                        self.x_regs.write(rd, old);
                        self.next_inst32();
                    },
                    0b111 => { // CSRRCI
                        let old = self.csr.read(csr, self.level)?;
                        if uimm != 0 {
                            self.csr.write(csr, old & !uimm, self.level)?;
                        }
                        self.x_regs.write(rd, old);
                        self.next_inst32();
                    },
                    _ => return Err(Exception::IllegalInstruction),
                }
            },
            0x1b => { // I-Type W instructions (RV64I)
                let (rd, rs1, funct3, imm_raw) = decode32_i(inst);
                let rs1_val = self.x_regs.read(rs1);
                let result = match funct3 {
                    0b000 => { // ADDIW
                        let imm = sign_extend(imm_raw as u64, 12);
                        let val = rs1_val.wrapping_add(imm);
                        ((val as i32) as i64) as u64
                    }
                    0b001 => { // SLLIW
                        let shamt = (imm_raw & 0x1f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        if funct7 != 0b0000000 {
                            return Err(Exception::IllegalInstruction);
                        }
                        ((rs1_val as u32) << shamt) as i32 as i64 as u64
                    }
                    0b101 => {
                        let shamt = (imm_raw & 0x1f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        match funct7 {
                            0b0000000 => ((rs1_val as u32) >> shamt) as u64,                   // SRLIW
                            0b0100000 => ((rs1_val as i32) >> shamt) as i64 as u64,           // SRAIW
                            _ => return Err(Exception::IllegalInstruction),
                        }
                    }
                    _ => return Err(Exception::IllegalInstruction),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x3b => { // R-Type W operations (RV64I)
                let (rd, rs1, rs2, funct3, funct7) = decode32_r(inst);
                let val1 = self.x_regs.read(rs1) as u32;
                let val2 = self.x_regs.read(rs2) as u32;
                let result = match (funct3, funct7) {
                    (0b000, 0b0000000) => (val1.wrapping_add(val2)) as i32 as i64 as u64, // ADDW
                    (0b000, 0b0100000) => (val1.wrapping_sub(val2)) as i32 as i64 as u64, // SUBW
                    (0b001, 0b0000000) => (val1 << (val2 & 0x1f)) as i32 as i64 as u64,   // SLLW
                    (0b101, 0b0000000) => (val1 >> (val2 & 0x1f)) as i32 as i64 as u64,  // SRLW
                    (0b101, 0b0100000) => ((val1 as i32) >> (val2 & 0x1f)) as i64 as u64, // SRAW
                    _ => return Err(Exception::IllegalInstruction),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            }
            _ => {}
        }

        Ok(())
    }

    pub fn execute16(&mut self, inst: u16) -> Result<(), Exception> {
        if inst == 0 {
            return Err(Exception::IllegalInstruction);
        }

        let opcode = inst & 0b11;


        Ok(())
    }

    pub fn dump(&self) {
        for i in 0..REGS_COUNT {
            println!("x{:02} = 0x{:016x}", i, self.x_regs.read(i as u64));
        }
    }
}

