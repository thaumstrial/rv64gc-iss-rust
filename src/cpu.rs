use crate::bus::{Bus, DRAM_BASE};
use crate::csr::{CSRs, MCAUSE, MEPC, MTVAL, MTVEC};
use crate::exception::{Exception, Trap};

pub const REGS_COUNT: usize = 32;
pub const BYTE: u8 = 8;
pub const HALF_WORD: u8 = 16;
pub const WORD: u8 = 32;
pub const DOUBLE_WORD: u8 = 64;
pub const PRIV_U: u8 = 0;
pub const PRIV_S: u8 = 1;
pub const PRIV_M: u8 = 3;
pub fn sign_extend(val: u64, bits: u8) -> u64 {
    let shift = 64 - bits;
    ((val << shift) as i64 >> shift) as u64
}
pub fn decode_r(inst: u32) -> (u64, u64, u64, u8, u8) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let rs2 = ((inst >> 20) & 0x1f) as u64;
    let funct7 = ((inst >> 25) & 0x7f) as u8;
    (rd, rs1, rs2, funct3, funct7)
}

// sign-extend in execution
pub fn decode_i(inst: u32) -> (u64, u64, u8, u16) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let imm = ((inst >> 20) & 0xfff) as u16;
    (rd, rs1, funct3, imm)
}
// sign-extend in execution
pub fn decode_s(inst: u32) -> (u64, u64, u8, u16) {
    let imm = (((inst >> 25) << 5) | ((inst >> 7) & 0x1f)) as u16;
    let funct3 = ((inst >> 12) & 0x7) as u8;
    let rs1 = ((inst >> 15) & 0x1f) as u64;
    let rs2 = ((inst >> 20) & 0x1f) as u64;
    (rs1, rs2, funct3, imm)
}

pub fn decode_b(inst: u32) -> (u64, u64, u8, u64) {
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

pub fn decode_u(inst: u32) -> (u64, u64) {
    let rd = ((inst >> 7) & 0x1f) as u64;
    let imm = (inst & 0xfffff000) as u64;
    (rd, imm)
}

pub fn decode_j(inst: u32) -> (u64, u64) {
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

pub fn decode_cr(inst: u16) -> (u64, u64, u8) {
    let rd_rs1 = ((inst >> 7) & 0x1f) as u64;
    let rs2 = ((inst >> 2) & 0x1f) as u64;
    let funct4 = ((inst >> 12) & 0xf) as u8;
    (rs2, rd_rs1, funct4)
}
pub fn decode_ci(inst: u16) -> (u64, u32) {
    let rd_rs1 = ((inst >> 7) & 0x1f) as u64;
    let raw_imm = (((inst >> 2) & 0x1f) | ((inst >> 12) & 0x1) << 6) as u32;
    (rd_rs1, raw_imm)
}
pub fn decode_css(inst: u16) -> (u64, u8) {
    let rs2 = ((inst >> 2) & 0x1f) as u64;
    let raw_imm = ((inst >> 7) & 0x3f) as u8;
    (rs2, raw_imm)
}
pub fn decode_ciw(inst: u16) -> (u64, u8) {
    let rd_c = 8 + ((inst >> 2) & 0x7) as u64;
    let raw_imm = ((inst >> 5) & 0xff) as u8;
    (rd_c, raw_imm)
}
pub fn decode_cl(inst: u16) -> (u64, u64, u8) {
    let rd_c = 8 + ((inst >> 2) & 0x7) as u64;
    let rs1_c = 8 + ((inst >> 7) & 0x7) as u64;
    let raw_imm = (((inst >> 5) & 0x3) | (((inst >> 10) & 0x7) << 2)) as u8;
    (rd_c, rs1_c, raw_imm)
}
pub fn decode_cs(inst: u16) -> (u64, u64, u8) {
    let rs2_c = 8 + ((inst >> 2) & 0x7) as u64;
    let rs1_c = 8 + ((inst >> 7) & 0x7) as u64;
    let imm = (((inst >> 5) & 0x3) | ((inst >> 10) & 0x7) << 2) as u8;
    (rs1_c, rs2_c, imm)
}
pub fn decode_ca(inst: u16) -> (u64, u64, u8, u8) {
    let rs2_c = 8 + ((inst >> 2) & 0x7) as u64;
    let funct2 = ((inst >> 5) & 0x3) as u8;
    let rd_rs1_c = 8 + ((inst >> 7) & 0x7) as u64;
    let funct6 = ((inst >> 10) & 0x3f) as u8;
    (rs2_c, rd_rs1_c, funct2, funct6)
}
pub fn decode_cb(inst: u16) -> (u64, u16, u16) {
    let rd_rs1_c = 8 + ((inst >> 7) & 0x7) as u64;
    let offset_r = (inst >> 2) & 0x1f;
    let offset_l = (inst >> 10) & 0x7;
    (rd_rs1_c, offset_l, offset_r)
}

pub fn decode_cj(inst: u16) -> u64 {
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
            level: PRIV_M,
        }
    }

    pub fn handle_trap(&self, trap: Trap) {
        if self.level <= PRIV_S {

        } else {
            let cause = match trap {
                Trap::Exception(e) => {
                    0 << 63 | (e as u64)
                }
                Trap::Interrupt(i) => {
                    1 << 63 | (i as u64)
                }
            };
            self.csr.0[MCAUSE] = cause;
            self.csr.0[MEPC] = self.pc;
            self.csr.0[MTVAL] = 0;
        }
    }

    pub fn run(&mut self) {
        loop {
            if let Err(e) = self.step() {
                self.handle_trap(e);
            }
        }
    }

    pub fn step(&mut self) -> Result<(), Trap> {
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

    pub fn execute32(&mut self, inst: u32) -> Result<(), Trap> {
        let opcode = inst & 0x7f;
        match opcode {
            0x37 => { // LUI
                let (rd, imm) = decode_u(inst);
                self.x_regs.write(rd, imm);
                self.next_inst32();
            },
            0x17 => { // AUIPC
                let (rd, imm) = decode_u(inst);
                self.x_regs.write(rd, self.pc.wrapping_add(imm));
                self.next_inst32();
            },
            0x6f => { // JAL
                let (rd, offset) = decode_j(inst);
                self.x_regs.write(rd, self.pc.wrapping_add(4));
                self.jump_inst(offset);
            },
            0x67 => { // JALR
                let (rd, rs1, funct3, imm) = decode_i(inst);
                if funct3 != 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }
                let target = self.x_regs.read(rs1)
                    .wrapping_add(sign_extend(imm as u64, 12)) & !1;
                self.x_regs.write(rd, self.pc.wrapping_add(4));
                self.pc = target;
            },
            0x63 => { // branch
                let (rs1, rs2, funct3, offset) = decode_b(inst);
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
                let (rd, rs1, funct3, imm) = decode_i(inst);
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
                let (rs1, rs2, funct3, imm) = decode_s(inst);
                let offset = sign_extend(imm as u64, 12);
                let addr = self.x_regs.read(rs1).wrapping_add(offset);
                let val = self.x_regs.read(rs2);
                match funct3 {
                    0b000 => self.bus.write(addr, val, BYTE)?, // SB
                    0b001 => self.bus.write(addr, val, HALF_WORD)?, // SH
                    0b010 => self.bus.write(addr, val, WORD)?, // SW
                    0b011 => self.bus.write(addr, val, DOUBLE_WORD)?, // SD (RV64I)
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                }
                self.next_inst32();
            },
            0x13 => { // I-type ALU
                let (rd, rs1, funct3, imm_raw) = decode_i(inst);
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
                            return Err(Trap::Exception(Exception::IllegalInstruction));
                        }
                        rs1_val << shamt
                    }
                    0b101 => {
                        let shamt = (imm_raw & 0x3f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        match funct7 {
                            0b0000000 => rs1_val >> shamt,                    // SRLI
                            0b0100000 => ((rs1_val as i64) >> shamt) as u64, // SRAI
                            _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                        }
                    },
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                };

                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x33 => { // R-Type ALU
                let (rd, rs1, rs2, funct3, funct7) = decode_r(inst);
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
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x0f => { // FENCE, FENCE.TSO, PAUSE
                // ignored in the single-core implementation
                let (_, _, funct3, _) = decode_i(inst);
                match funct3 {
                    0b000 => self.next_inst32(),  // FENCE
                    0b001 => self.next_inst32(),  // FENCE.I (RV32/RV64 Fiencei)
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                }
            },
            0x73 => { // SYSTEM
                let (rd, rs1, funct3, imm) = decode_i(inst);
                let csr = imm;
                let uimm = rs1;

                match funct3 {
                    0b000 => {
                        if !(rd != 0 && rs1 == 0) {
                            return Err(Trap::Exception(Exception::IllegalInstruction));
                        }
                        return match imm {
                            0x000 => { // ECALL
                                match self.level {
                                    0b00 => {
                                        Err(Trap::Exception(Exception::EnvironmentCallFromUMode))
                                    },
                                    0b01 => {
                                        Err(Trap::Exception(Exception::EnvironmentCallFromSMode))
                                    },
                                    0b11 => {
                                        Err(Trap::Exception(Exception::EnvironmentCallFromMMode))
                                    },
                                    _ => { unreachable!() },
                                }
                            },
                            0x001 => { // EBREAK
                                Err(Trap::Exception(Exception::Breakpoint))
                            },
                            _ => Err(Trap::Exception(Exception::IllegalInstruction)),
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
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                }
            },
            0x1b => { // I-Type W instructions (RV64I)
                let (rd, rs1, funct3, imm_raw) = decode_i(inst);
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
                            return Err(Trap::Exception(Exception::IllegalInstruction));
                        }
                        ((rs1_val as u32) << shamt) as i32 as i64 as u64
                    }
                    0b101 => {
                        let shamt = (imm_raw & 0x1f) as u8;
                        let funct7 = (inst >> 25) & 0x7f;
                        match funct7 {
                            0b0000000 => ((rs1_val as u32) >> shamt) as u64,                   // SRLIW
                            0b0100000 => ((rs1_val as i32) >> shamt) as i64 as u64,           // SRAIW
                            _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                        }
                    }
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            },
            0x3b => { // R-Type W operations (RV64I)
                let (rd, rs1, rs2, funct3, funct7) = decode_r(inst);
                let val1 = self.x_regs.read(rs1) as u32;
                let val2 = self.x_regs.read(rs2) as u32;
                let result = match (funct3, funct7) {
                    (0b000, 0b0000000) => (val1.wrapping_add(val2)) as i32 as i64 as u64, // ADDW
                    (0b000, 0b0100000) => (val1.wrapping_sub(val2)) as i32 as i64 as u64, // SUBW
                    (0b001, 0b0000000) => (val1 << (val2 & 0x1f)) as i32 as i64 as u64,   // SLLW
                    (0b101, 0b0000000) => (val1 >> (val2 & 0x1f)) as i32 as i64 as u64,  // SRLW
                    (0b101, 0b0100000) => ((val1 as i32) >> (val2 & 0x1f)) as i64 as u64, // SRAW
                    _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                };
                self.x_regs.write(rd, result);
                self.next_inst32();
            }
            _ => {}
        }

        Ok(())
    }

    pub fn execute16(&mut self, inst: u16) -> Result<(), Trap> {
        if inst == 0 {
            return Err(Trap::Exception(Exception::IllegalInstruction));
        }

        let opcode = inst & 0b11;
        let funct3 = (inst >> 13) & 0b111;
        match (opcode, funct3) {
            (0b00, 0b000) => { // C.ADDI4SPN
                let (rd_c, raw_imm) = decode_ciw(inst);
                if raw_imm == 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }
                let offset = (
                    (raw_imm & 0x1) << 3
                    | ((raw_imm >> 1) & 0x1) << 2
                    | ((raw_imm >> 2) & 0xf) << 6
                    | ((raw_imm >> 6) & 0x3) << 4
                ) as u64;
                let val = self.x_regs.read(2).wrapping_add(offset);
                self.x_regs.write(rd_c, val);
                self.next_inst16();
            },
            (0b00, 0b001) => { // C.FLD

            },
            (0b00, 0b010) => { // C.LW
                let (rd_c, rs1_c, raw_imm) = decode_cl(inst);
                let offset = (
                    (raw_imm & 0x1) << 6
                    | ((raw_imm >> 1) & 0x1) << 2
                    | ((raw_imm >> 2) & 0x3) << 3
                ) as u64;
                let addr = self.x_regs.read(rs1_c).wrapping_add(offset);
                let val = self.bus.read(addr, WORD)?;
                self.x_regs.write(rd_c, val);
                self.next_inst16();
            },
            (0b00, 0b011) => { // C.LD
               let (rd_c, rs1_c, raw_imm) = decode_cl(inst);
                let offset = (
                     (raw_imm & 0x3) << 6
                     | ((raw_imm >> 2) & 0x3) << 3
                ) as u64;
                let addr = self.x_regs.read(rs1_c).wrapping_add(offset);
                let val = self.bus.read(addr, DOUBLE_WORD)?;
                self.x_regs.write(rd_c, val);
                self.next_inst16();
            },
            (0b00, 0b101) => { // C.FSD

            },
            (0b00, 0b110) => { // C.SW
                let (rs1_c, rs2_c, raw_imm) = decode_cs(inst);
                let offset = (
                    (raw_imm & 0x1) << 6
                    | ((raw_imm >> 1) & 0x1) << 2
                    | ((raw_imm >> 2) & 0x3) << 3
                ) as u64;
                let addr = self.x_regs.read(rs1_c).wrapping_add(offset);
                let val = self.x_regs.read(rs2_c);
                self.bus.write(addr, val, WORD)?;
                self.next_inst16();
            },
            (0b00, 0b111) => { // C.SD
                let (rs1_c, rs2_c, raw_imm) = decode_cs(inst);
                let offset = (
                    (raw_imm & 0x3) << 6
                    | ((raw_imm >> 2) & 0x3) << 3
                ) as u64;
                let addr = self.x_regs.read(rs1_c).wrapping_add(offset);
                let val = self.x_regs.read(rs2_c);
                self.bus.write(addr, val, DOUBLE_WORD)?;
                self.next_inst16();
            },
            (0b01, 0b000) => { // C.NOP
                let (rd, raw_imm) = decode_ci(inst);
                if rd != 0 { // C.ADDI
                    let nzimm = sign_extend(raw_imm as u64, 6);
                    let val = self.x_regs.read(rd).wrapping_add(nzimm);
                    self.x_regs.write(rd, val);
                }
                self.next_inst16();
            },
            (0b01, 0b001) => { // C.ADDIW
                let (rd, raw_imm) = decode_ci(inst);
                if rd == 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }

                let nzimm = sign_extend(raw_imm as u64, 6);
                let val = self.x_regs.read(rd).wrapping_add(nzimm) as i32 as i64 as u64;
                self.x_regs.write(rd, val);
                self.next_inst16();
            },
            (0b01, 0b010) => { // C.LI
                let (rd, raw_imm) = decode_ci(inst);
                if rd != 0 {
                    let imm = sign_extend(raw_imm as u64, 6);
                    self.x_regs.write(rd, imm);
                }
                self.next_inst16();
            },
            (0b01, 0b011) => {
                let (rd, raw_imm) = decode_ci(inst);
                if rd == 2 { // C.ADDI16SP
                    let nzimm = sign_extend((
                        (raw_imm & 0x1) << 5
                        | ((raw_imm >> 1) & 0x3) << 7
                        | ((raw_imm >> 4) & 0x1) << 6
                        | ((raw_imm >> 5) & 0x1) << 4
                        | ((raw_imm >> 6) & 0x1) << 9
                    ) as u64, 10);
                    if nzimm == 0 {
                        return Err(Trap::Exception(Exception::IllegalInstruction));
                    }
                    let val = self.x_regs.read(2).wrapping_add(nzimm);
                    self.x_regs.write(2, val);
                } else if rd != 0 { // C.LUI
                    let nzimm = sign_extend(
                        (
                            ((raw_imm & 0x1f) << 12)
                            | ((raw_imm >> 5) & 0x1) << 17
                        ) as u64,
                        18,
                    );
                    self.x_regs.write(rd, nzimm);
                }
                self.next_inst16();
            },
            (0b01, 0b100) => {
                let funct2 = (inst >> 10) & 0b11;
                match funct2 {
                    0b00 => { // C.SRLI
                        let (rd_c, offset_l, offset_r) = decode_cb(inst);
                        let shamt = offset_r | (((offset_l >> 2) & 0x1) << 5);
                        if shamt != 0 {
                            let val = self.x_regs.read(rd_c) >> shamt;
                            self.x_regs.write(rd_c, val);
                        }
                    },
                    0b01 => { // C.SRAI
                        let (rd_c, offset_l, offset_r) = decode_cb(inst);
                        let shamt = offset_r | (((offset_l >> 2) & 0x1) << 5);
                        if shamt != 0 {
                            let val = (self.x_regs.read(rd_c) as i64) >> shamt;
                            self.x_regs.write(rd_c, val as u64);
                        }
                    },
                    0b10 => { // C.ANDI
                        let (rd_c, raw_imm) = decode_ci(inst);
                        let imm = sign_extend(raw_imm as u64, 6);
                        let val = self.x_regs.read(rd_c) & imm;
                        self.x_regs.write(rd_c, val);
                    },
                    0b11 => {
                        let func3 = ((inst >> 5) & 0b11) | ((inst >> 2) & 0b100);
                        let (rs2_c, rd_c, _, _) = decode_ca(inst);
                        let val1 = self.x_regs.read(rd_c);
                        let val2 = self.x_regs.read(rs2_c);

                        let result = match func3 {
                            0b000 => val1.wrapping_sub(val2),                     // C.SUB
                            0b001 => val1 ^ val2,                                 // C.XOR
                            0b010 => val1 | val2,                                 // C.OR
                            0b011 => val1 & val2,                                 // C.AND
                            0b100 => {                                             // C.SUBW
                                let v1 = val1 as u32;
                                let v2 = val2 as u32;
                                v1.wrapping_sub(v2) as i32 as i64 as u64
                            }
                            0b101 => {                                             // C.ADDW
                                let v1 = val1 as u32;
                                let v2 = val2 as u32;
                                v1.wrapping_add(v2) as i32 as i64 as u64
                            }
                            _ => return Err(Trap::Exception(Exception::IllegalInstruction)),
                        };

                        self.x_regs.write(rd_c, result);
                    },
                    _ => {unreachable!()}
                }
                self.next_inst16();
            },
            (0b01, 0b101) => { // C.J
                let imm = decode_cj(inst);
                self.jump_inst(imm);
            },
            (0b01, 0b110) => { // C.BEQZ
                let (rs1_c, offset_l, offset_r) = decode_cb(inst);
                let offset = sign_extend(
                    (
                            (offset_r & 0x1) << 5
                            | ((offset_r >> 1) & 0x3) << 1
                            | ((offset_r >> 3) & 0x3) << 6
                            | (offset_l & 0x3) << 3
                            | ((offset_l >> 2) & 0x1) << 8
                        ) as u64,
                    9
                );
                if self.x_regs.read(rs1_c) == 0 {
                    self.jump_inst(offset);
                } else {
                    self.next_inst16();
                }
            },
            (0b01, 0b111) => { // C.BNEZ
                let (rs1_c, offset_l, offset_r) = decode_cb(inst);
                let offset = sign_extend(
                    (
                            (offset_r & 0x1) << 5
                            | ((offset_r >> 1) & 0x3) << 1
                            | ((offset_r >> 3) & 0x3) << 6
                            | (offset_l & 0x3) << 3
                            | ((offset_l >> 2) & 0x1) << 8
                        ) as u64,
                    9
                );
                if self.x_regs.read(rs1_c) != 0 {
                    self.jump_inst(offset);
                } else {
                    self.next_inst16();
                }
            },
            (0b10, 0b000) => { // C.SLLI
                let (rd_c, raw_imm) = decode_ci(inst);
                let shamt = raw_imm as u8;
                if shamt != 0 && rd_c != 0 {
                    let val = self.x_regs.read(rd_c) << shamt;
                    self.x_regs.write(rd_c, val);
                }
                self.next_inst16();
            },
            (0b10, 0b001) => { // C.FLDSP

            },
            (0b10, 0b010) => { // C.LWSP
                let (rd, raw_imm) = decode_ci(inst);
                if rd == 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }
                let offset = sign_extend(
                    (
                        (raw_imm & 0x3) << 6
                        | ((raw_imm >> 2) & 0x7) << 2
                        | ((raw_imm >> 5) & 0x1) << 5
                    ) as u64,
                    8,
                );
                let addr = self.x_regs.read(2).wrapping_add(offset);
                let val = self.bus.read(addr, WORD)?;
                self.x_regs.write(rd, val);
                self.next_inst16();
            },
            (0b10, 0b011) => { // C.LDSP
                let (rd, raw_imm) = decode_ci(inst);
                if rd == 0 {
                    return Err(Trap::Exception(Exception::IllegalInstruction));
                }
                let offset = sign_extend(
                    (
                        (raw_imm & 0x7) << 6
                        | ((raw_imm >> 3) & 0x3) << 3
                        | ((raw_imm >> 5) & 0x1) << 5
                    ) as u64,
                    9,
                );
                let addr = self.x_regs.read(2).wrapping_add(offset);
                let val = self.bus.read(addr, DOUBLE_WORD)?;
                self.x_regs.write(rd, val);
                self.next_inst16();
            },
            (0b10, 0b100) => {
                let (rs2, rs1, funct4) = decode_cr(inst);
                match funct4 & 0x1 {
                    0 => {
                        match rs1  {
                            0 => { // C.JR
                                if rs1 == 0 {
                                    return Err(Trap::Exception(Exception::IllegalInstruction));
                                }
                                self.jump_inst(self.x_regs.read(rs1));
                            },
                            _ => { // C.MV
                                if rs2 == 0 {
                                    return Err(Trap::Exception(Exception::IllegalInstruction));
                                }
                                if rs1 != 0 {
                                    self.x_regs.write(rs1, self.x_regs.read(rs2));
                                }
                                self.next_inst16();
                            }
                        }
                    },
                    1 => {
                        if rs1 == 0 { // C.EBREAK
                            if rs2 != 0 {
                                return Err(Trap::Exception(Exception::IllegalInstruction));
                            }
                            return Err(Trap::Exception(Exception::Breakpoint));
                        } else {
                            if rs2 == 0 { // C.JALR
                                let offset = self.x_regs.read(rs1);
                                self.x_regs.write(1, self.pc.wrapping_add(2));
                                self.jump_inst(offset);
                            } else { // C.ADD
                                if rs1 != 0 {
                                    let val = self.x_regs.read(rs1).wrapping_add(self.x_regs.read(rs2));
                                    self.x_regs.write(rs1, val);
                                }
                                self.next_inst16();
                            }
                        }
                    },
                    _ => {unreachable!()}
                }
            },
            (0b10, 0b101) => { // C.FSDSP

            },
            (0b10, 0b110) => { // C.SWSP
                let (rs2, raw_imm) = decode_css(inst);
                let offset = sign_extend(
                    (
                        (raw_imm & 0x3) << 6
                        | ((raw_imm >> 2) & 0xf) << 2
                    ) as u64,
                    8,
                );
                let addr = self.x_regs.read(2).wrapping_add(offset);
                let val = self.x_regs.read(rs2);
                self.bus.write(addr, val, WORD)?;
                self.next_inst16();
            },
            (0b10, 0b111) => { // C.SDSP
                let (rs2, raw_imm) = decode_css(inst);
                let offset = sign_extend(
                    (
                        (raw_imm & 0x7) << 6
                        | ((raw_imm >> 3) & 0x3) << 3
                    ) as u64,
                    9,
                );
                let addr = self.x_regs.read(2).wrapping_add(offset);
                let val = self.x_regs.read(rs2);
                self.bus.write(addr, val, DOUBLE_WORD)?;
                self.next_inst16();
            },
            (_, _) => {
                unreachable!()
            }
        }

        Ok(())
    }

    pub fn dump(&self) {
        for i in 0..REGS_COUNT {
            println!("x{:02} = 0x{:016x}", i, self.x_regs.read(i as u64));
        }
    }
}

