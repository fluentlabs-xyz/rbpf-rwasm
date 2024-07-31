// Copyright 2016 6WIND S.A. <quentin.monnet@6wind.com>
//
// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This module contains error and result types

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use {
    crate::{elf::ElfError, memory_region::AccessType, verifier::VerifierError},
    core::fmt,
};

#[derive(Debug)]
pub enum MemoryError {
    InvalidInput,
    WriteFailed,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MemoryError::InvalidInput => write!(f, "aligned memory fill_write failed"),
            MemoryError::WriteFailed => write!(f, "aligned memory write failed"),
        }
    }
}


pub trait MyError: Debug + Display {
    fn source(&self) -> Option<&(dyn MyError + 'static)> {
        None
    }
}


pub struct MyErrorWrapper {
    pub inner: Box<dyn MyError>,
}

impl fmt::Debug for MyErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl fmt::Display for MyErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl MyError for MyErrorWrapper {

}

impl From<EbpfError> for MyErrorWrapper {
    fn from(error: EbpfError) -> Self {
        MyErrorWrapper {
            inner: Box::new(error)
        }
    }
}


#[derive(Debug)]
pub struct InvalidInputError {
    message: String,
}

impl InvalidInputError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for InvalidInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid input: {}", self.message)
    }
}

impl MyError for InvalidInputError {}


#[derive(Debug)]
pub struct WriteFailedError {
    message: String,
}

impl WriteFailedError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for WriteFailedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Write operation failed: {}", self.message)
    }
}

impl MyError for WriteFailedError {}
#[derive(Debug)]
pub struct CustomError {
    description: String,
}

impl CustomError {
    pub fn new(description: impl Into<String>) -> Self {
        Self { description: description.into() }
    }
}

#[derive(Debug)]
pub struct SimpleError {
    message: String,
}

impl Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl MyError for SimpleError {}


impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}


impl MyError for CustomError {}

impl From<fmt::Error> for CustomError {
    fn from(_: fmt::Error) -> Self {
        CustomError::new("write error")
    }
}

#[derive(Debug)]
pub struct FormatError {
    message: String,
}

impl FormatError {
    pub fn new(msg: String) -> Self {
        FormatError { message: msg }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl MyError for FormatError {}


pub trait MyWrite: fmt::Write {
    fn write(&mut self, buf: &[u8]) -> Result<usize, CustomError>;
    fn flush(&mut self) -> Result<(), CustomError>;
}

impl From<fmt::Error> for Box<dyn MyError> {
    fn from(err: fmt::Error) -> Self {
        Box::new(FormatError::new(err.to_string()))
    }
}

pub struct MyBuffer {
    pub buf: Vec<u8>,
}

impl fmt::Write for MyBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

impl MyWrite for MyBuffer {
    fn write(&mut self, buf: &[u8]) -> Result<usize, CustomError> {
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), CustomError> {
        Ok(())
    }
}

/// Error definitions
#[derive(Debug)]
#[repr(u64)] // discriminant size, used in emit_exception_kind in JIT
pub enum EbpfError {
    /// ELF error
    // #[error("ELF error: {0}")]
    ElfError(ElfError),
    /// Function was already registered
    // #[error("function #{0} was already registered")]
    FunctionAlreadyRegistered(usize),
    /// Exceeded max BPF to BPF call depth
    // #[error("exceeded max BPF to BPF call depth")]
    CallDepthExceeded,
    /// Attempt to exit from root call frame
    // #[error("attempted to exit root call frame")]
    ExitRootCallFrame,
    /// Divide by zero"
    //#[error("divide by zero at BPF instruction")]
    DivideByZero,
    /// Divide overflow
    //#[error("division overflow at BPF instruction")]
    DivideOverflow,
    /// Exceeded max instructions allowed
    //#[error("attempted to execute past the end of the text segment at BPF instruction")]
    ExecutionOverrun,
    /// Attempt to call to an address outside the text segment
    //#[error("callx attempted to call outside of the text segment")]
    CallOutsideTextSegment,
    /// Exceeded max instructions allowed
    //#[error("exceeded CUs meter at BPF instruction")]
    ExceededMaxInstructions,
    /// Program has not been JIT-compiled
    //#[error("program has not been JIT-compiled")]
    JitNotCompiled,
    /// Invalid virtual address
    //#[error("invalid virtual address {0:x?}")]
    InvalidVirtualAddress(u64),
    /// Memory region index or virtual address space is invalid
    //#[error("Invalid memory region at index {0}")]
    InvalidMemoryRegion(usize),
    /// Access violation (general)
    //#[error("Access violation in {3} section at address {1:#x} of size {2:?}")]
    AccessViolation(AccessType, u64, u64, &'static str),
    /// Access violation (stack specific)
    //#[error("Access violation in stack frame {3} at address {1:#x} of size {2:?}")]
    StackAccessViolation(AccessType, u64, u64, i64),
    /// Invalid instruction
    //#[error("invalid BPF instruction")]
    InvalidInstruction,
    /// Unsupported instruction
    //#[error("unsupported BPF instruction")]
    UnsupportedInstruction,
    /// Compilation is too big to fit
    //#[error("Compilation exhausted text segment at BPF instruction {0}")]
    ExhaustedTextSegment(usize),
    /// Libc function call returned an error
    //#[error("Libc calling {0} {1:?} returned error code {2}")]
    LibcInvocationFailed(&'static str, Vec<String>, i32),
    /// Verifier error
    //#[error("Verifier error: {0}")]
    VerifierError(VerifierError),
    /// Syscall error
    //#[error("Syscall error: {0}")]
    SyscallError(Box<dyn MyError>),
}

impl From<ElfError> for EbpfError {
    fn from(error: ElfError) -> Self {
        EbpfError::ElfError(error)
    }
}

impl fmt::Display for EbpfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EbpfError::ElfError(err) => write!(f, "ELF error: {}", err),
            EbpfError::FunctionAlreadyRegistered(index) => write!(f, "Function #{} was already registered", index),
            EbpfError::CallDepthExceeded => write!(f, "Exceeded max BPF to BPF call depth"),
            EbpfError::ExitRootCallFrame => write!(f, "Attempted to exit root call frame"),
            EbpfError::DivideByZero => write!(f, "Divide by zero at BPF instruction"),
            EbpfError::DivideOverflow => write!(f, "Division overflow at BPF instruction"),
            EbpfError::ExecutionOverrun => write!(f, "Attempted to execute past the end of the text segment at BPF instruction"),
            EbpfError::CallOutsideTextSegment => write!(f, "CallX attempted to call outside of the text segment"),
            EbpfError::ExceededMaxInstructions => write!(f, "Exceeded CUs meter at BPF instruction"),
            EbpfError::JitNotCompiled => write!(f, "Program has not been JIT-compiled"),
            EbpfError::InvalidVirtualAddress(address) => write!(f, "Invalid virtual address {:#x}", address),
            EbpfError::InvalidMemoryRegion(index) => write!(f, "Invalid memory region at index {}", index),
            EbpfError::AccessViolation(access_type, address, size, section) => write!(f, "Access violation in {} section at address {:#x} of size {}", section, address, size),
            EbpfError::StackAccessViolation(access_type, address, size, frame) => write!(f, "Access violation in stack frame {} at address {:#x} of size {}", frame, address, size),
            EbpfError::InvalidInstruction => write!(f, "Invalid BPF instruction"),
            EbpfError::UnsupportedInstruction => write!(f, "Unsupported BPF instruction"),
            EbpfError::ExhaustedTextSegment(instr) => write!(f, "Compilation exhausted text segment at BPF instruction {}", instr),
            EbpfError::LibcInvocationFailed(func, args, err_code) => {
                let args_joined = args.join(", ");
                write!(f, "Libc calling {}({}) returned error code {}", func, args_joined, err_code)
            },
            EbpfError::VerifierError(err) => write!(f, "Verifier error: {}", err),
            EbpfError::SyscallError(boxed_err) => write!(f, "Syscall error: {}", boxed_err),
        }
    }
}


impl MyError for EbpfError {
    fn source(&self) -> Option<&(dyn MyError + 'static)> {
        match self {
            EbpfError::SyscallError(ref err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// Same as `Result` but provides a stable memory layout
#[derive(Debug)]
#[repr(C, u64)]
pub enum StableResult<T, E> {
    /// Success
    Ok(T),
    /// Failure
    Err(E),
}

impl<T: fmt::Debug, E: fmt::Debug> StableResult<T, E> {
    /// `true` if `Ok`
    pub fn is_ok(&self) -> bool {
        match self {
            Self::Ok(_) => true,
            Self::Err(_) => false,
        }
    }

    /// `true` if `Err`
    pub fn is_err(&self) -> bool {
        match self {
            Self::Ok(_) => false,
            Self::Err(_) => true,
        }
    }

    /// Returns the inner value if `Ok`, panics otherwise
    pub fn unwrap(self) -> T {
        match self {
            Self::Ok(value) => value,
            Self::Err(error) => panic!("unwrap {:?}", error),
        }
    }

    /// Returns the inner error if `Err`, panics otherwise
    pub fn unwrap_err(self) -> E {
        match self {
            Self::Ok(value) => panic!("unwrap_err {:?}", value),
            Self::Err(error) => error,
        }
    }

    /// Maps ok values, leaving error values untouched
    pub fn map<U, O: FnOnce(T) -> U>(self, op: O) -> StableResult<U, E> {
        match self {
            Self::Ok(value) => StableResult::<U, E>::Ok(op(value)),
            Self::Err(error) => StableResult::<U, E>::Err(error),
        }
    }

    /// Maps error values, leaving ok values untouched
    pub fn map_err<F, O: FnOnce(E) -> F>(self, op: O) -> StableResult<T, F> {
        match self {
            Self::Ok(value) => StableResult::<T, F>::Ok(value),
            Self::Err(error) => StableResult::<T, F>::Err(op(error)),
        }
    }

    #[cfg_attr(
        any(
            not(feature = "jit"),
            target_os = "windows",
            not(target_arch = "x86_64")
        ),
        allow(dead_code)
    )]
    pub(crate) fn discriminant(&self) -> u64 {
        unsafe { *core::ptr::addr_of!(*self).cast::<u64>() }
    }
}

impl<T, E> From<StableResult<T, E>> for Result<T, E> {
    fn from(result: StableResult<T, E>) -> Self {
        match result {
            StableResult::Ok(value) => Ok(value),
            StableResult::Err(value) => Err(value),
        }
    }
}

impl<T, E> From<Result<T, E>> for StableResult<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(value) => Self::Ok(value),
            Err(value) => Self::Err(value),
        }
    }
}

/// Return value of programs and syscalls
pub type ProgramResult = StableResult<u64, EbpfError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_result_is_stable() {
        let ok = ProgramResult::Ok(42);
        assert_eq!(ok.discriminant(), 0);
        let err = ProgramResult::Err(EbpfError::JitNotCompiled);
        assert_eq!(err.discriminant(), 1);
    }
    use super::*;

    #[test]
    fn test_error_display() {
        let error = CustomError::new("test error");
        assert_eq!(format!("{}", error), "test error");
    }

    #[test]
    fn test_error_from_fmt() {
        let fmt_error = fmt::Error;
        let error: Box<dyn MyError> = fmt_error.into();
        assert_eq!(format!("{}", error), "Formatting error occurred");
    }
}
