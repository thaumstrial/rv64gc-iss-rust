pub enum Exception {
    IllegalInstruction,
    Breakpoint,
    LoadAccessFault,
    StoreAMOAccessFault,
    InstructionAccessFault,
    EnvironmentCallFromUMode,
    EnvironmentCallFromSMode,
    EnvironmentCallFromMMode
}