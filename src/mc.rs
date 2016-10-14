use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::io;
use std::fmt;

use encoding::{Encoding, DecoderTrap};
use encoding::all::WINDOWS_31J;

pub struct MemoryCard {
    /// Raw buffer containing the memory card image. Always `TOTAL_SIZE`
    /// long.
    raw: Box<[u8]>,
}

impl MemoryCard {
    pub fn from_path(path: &Path) -> Result<MemoryCard, Error> {
        let mut f = try!(File::open(path));

        MemoryCard::from_reader(&mut f)
    }

    pub fn from_reader(r: &mut Read) -> Result<MemoryCard, Error> {
        let mut buf = vec![0; TOTAL_SIZE];

        try!(r.read_exact(&mut buf));

        if try!(r.read(&mut[0])) > 0 {
            Err(Error::BadLength)
        } else {
            MemoryCard::from_buffer(buf)
        }
    }

    pub fn from_buffer(buf: Vec<u8>) -> Result<MemoryCard, Error> {
        if buf.len() != TOTAL_SIZE {
            return Err(Error::BadLength);
        }

        if &buf[0..2] != b"MC" {
            return Err(Error::BadFormat("missing 'MC' magic in header".into()));
        }

        if sector_checksum(&buf[0..128]).is_err() {
            return Err(Error::BadFormat("bad header checksum".into()));
        }

        let mc = MemoryCard {
            raw: buf.into_boxed_slice(),
        };

        if !mc.broken_sectors().is_empty() {
            // XXX add support for this when needed...
            return Err(Error::BadFormat("Card has broken sectors!".into()));
        }

        // Basic checks for directory incoherencies.
        for block in 1..16 {
            try!(mc.block(block).validate());
        }

        Ok(mc)
    }

    /// Return a reference to a single block
    pub fn block(&self, block: u8) -> Block {
        assert!(block >= 1 && block <= 15);

        let off = block as usize;

        let start = off * SECTOR_SIZE;
        let end = start + SECTOR_SIZE;

        let metadata = &self.raw[start..end];

        let start = off * BLOCK_SIZE;
        let end = start + BLOCK_SIZE;

        let data = &self.raw[start..end];

        Block::new(block, metadata, data)
    }

    pub fn broken_sectors(&self) -> Vec<BrokenSector> {

        let mut v = Vec::new();

        for s in 0..20 {
            let off = (16 + s) as usize;

            let start = off * SECTOR_SIZE;
            let end = (off + 1) * SECTOR_SIZE;

            let sector = &self.raw[start..end];

            let pos = read32(&sector[0..4]);

            if pos != 0xffffffff {
                // We have a broken sector
                v.push(
                    BrokenSector {
                        sector: pos,
                        replacement: 36 + s,
                        checksum: sector_checksum(sector),
                    });
            }
        }

        v
    }

    // Return a list with the first block of every file found. If
    // `deleted` is true then even deleted files are returned.
    pub fn files(&self, find_deleted: bool) -> Vec<u8> {
        let mut files = Vec::new();

        for b in 1..16 {
            let block = self.block(b);

            let (free, usage) = block.state();

            if (find_deleted || !free) && usage == BlockUsage::First {
                files.push(b);
            }
        }

        files
    }

    pub fn file_title(&self, first_block: u8) -> Result<String, Error> {
        let block = self.block(first_block);

        let (_, usage) = block.state();

        if usage != BlockUsage::First {
            return Err(Error::NotAFile(first_block));
        }

        let title = &block.data()[4..68];

        // The "encoding" crate doesn't support Shift JIS at the
        // moment, however as far as I know WINDOWS 31J should be
        // mostly the same thing. It seems to work with all the games
        // I've tried, at least.
        WINDOWS_31J.decode(title, DecoderTrap::Strict)
            .map(|mut s| {
                // Delete trailing zeroes.
                let eos = s.find('\0').unwrap_or(s.len());

                s.truncate(eos);

                s
            })
            .map_err(|_| Error::BadShiftJis)
    }

    pub fn file_blocks(&self, first_block: u8) -> Result<Vec<u8>, Error> {
        // We keep track of which blocks we've already used to detect
        // cyclical block references.
        let mut used_blocks = 0u16;

        let block = self.block(first_block);

        let (free_file, usage) = block.state();

        if usage != BlockUsage::First {
            return Err(Error::NotAFile(first_block));
        }

        // Follow the list
        let mut blocks = Vec::new();
        let mut next = Some(first_block);

        while let Some(b) = next {
            let mask = 1u16 << b;

            if used_blocks & mask != 0 {
                return Err(Error::CyclicalFile(b));
            }

            used_blocks |= mask;

            blocks.push(b);

            let block = self.block(b);

            if !free_file && block.is_free() {
                return Err(Error::FreeBlockFile(b));
            }

            next = block.next_block();
        }

        Ok(blocks)
    }
}

pub struct Block<'a> {
    /// This block's number (1...15)
    number: u8,
    /// Directory sector in block 0
    metadata: &'a [u8],
    /// Raw block data
    data: &'a [u8],
}

impl<'a> Block<'a> {
    fn new(number: u8, metadata: &'a [u8], data: &'a[u8]) -> Block<'a> {
        Block {
            number: number,
            metadata: metadata,
            data: data,
        }
    }

    /// Basic block sanity checks
    fn validate(&self) -> Result<(), Error> {

        let state = self.metadata[0];

        let free = state & 0xf0;
        let usage = state & 0xf;

        if (free != 0xa0 && free != 0x50) || usage > 3 {
            let err =
                format!("block {}: invalid or unsupported state (0x{:02x})",
                        self.number, state);

            return Err(Error::BadFormat(err));
        }

        Ok(())
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    pub fn number(&self) -> u8 {
        self.number
    }

    /// Return a pair `(free, usage)` containing the status of this
    /// block. `usage` doesn't really matter if `free` is true,
    /// however it could be useful in certain situations, such as
    /// trying to un-delete a file.
    pub fn state(&self) -> (bool, BlockUsage) {
        let state = self.metadata[0];

        let free =
            match state & 0xf0 {
                0xa0 => true,
                0x50 => false,
                _ => unreachable!(),
            };

        let usage =
            match state & 0xf {
                0 => BlockUsage::Unused,
                1 => BlockUsage::First,
                2 => BlockUsage::Middle,
                3 => BlockUsage::Last,
                _ => unreachable!(),
            };

        (free, usage)
    }

    pub fn is_free(&self) -> bool {
        let (free, _) = self.state();

        free
    }

    pub fn file_size(&self) -> u32 {
        read32(&self.metadata[4..8])
    }

    /// Return the next block number or `Err(raw_value)` if
    /// `raw_value` is not a valid block. Normally `raw_value` should
    /// always be `0xffff` when this is the last block.
    pub fn next_block_raw(&self) -> Result<u8, u16> {
        let next_block = read16(&self.metadata[8..10]);

        if next_block <= 14 {
            Ok(next_block as u8 + 1)
        } else {
            Err(next_block)
        }
    }

    /// Return the number of the next block in the file sequence
    /// (1...15), `None` otherwise.
    pub fn next_block(&self) -> Option<u8> {
        self.next_block_raw().ok()
    }

    /// Return the raw file name, a buffer of 20 bytes which should be
    /// a \0-terminated ASCII string.
    pub fn file_name_raw(&self) -> &[u8] {
        &self.metadata[10..31]
    }

    /// Return the file name as a human-readable String
    pub fn file_name(&self) -> String {
        let file_name = self.file_name_raw();

        let mut s = String::new();

        for &b in file_name {
            if b == 0 {
                break;
            }

            if b >= 0x20 && b <= 0x7e {
                // Printable ASCII
                s.push(b as char);
            } else {
                // Displa non-printable characters in hex. This
                // shoudn't happen since file names are supposed to be
                // ASCII in a standardized format.
                s = format!("{}\\x{:02x}", s, b)
            }
        }

        s
    }

    /// Return the metadata's checksum
    pub fn checksum(&self) -> Result<u8, u8> {
        sector_checksum(self.metadata)
    }
}

/// Convert 4 consecutive bytes into a 32bit little endian integer)
fn read32(b: &[u8]) -> u32 {
    let b0 = b[0] as u32;
    let b1 = b[1] as u32;
    let b2 = b[2] as u32;
    let b3 = b[3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Convert 2 consecutive bytes into a 16bit little endian integer)
fn read16(b: &[u8]) -> u16 {
    let b0 = b[0] as u16;
    let b1 = b[1] as u16;

    b0 | (b1 << 8)
}

/// Sector checksum, used to validate the integrity of a sector in
/// various places. Returns the 8bit checksum an `Ok` if it's valid,
/// in an `Error` otherwise.
fn sector_checksum(s: &[u8]) -> Result<u8, u8> {
    let c = s[0..127].iter().fold(0, |c, b| c ^ b);

    let expected = s[127];

    if c == expected {
        Ok(c)
    } else {
        Err(c)
    }
}

#[derive(Debug)]
pub enum Error {
    /// Invalid Shift JIS string
    BadShiftJis,
    /// A free block is referenced in a file
    FreeBlockFile(u8),
    /// A block is referenced twice in the same file
    CyclicalFile(u8),
    /// The block number provided is not the start of a file.
    NotAFile(u8),
    BadFormat(String),
    BadLength,
    IoError(io::Error),
}

impl ::std::convert::From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        if let io::ErrorKind::UnexpectedEof = e.kind() {
            Error::BadLength
        } else {
            Error::IoError(e)
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Error::BadShiftJis =>
                write!(f, "Invalid Shift JIS string"),
            &Error::FreeBlockFile(b) =>
                write!(f, "Free block {} used in a file", b),
            &Error::CyclicalFile(b) =>
                write!(f, "Cyclical reference of block {}", b),
            &Error::NotAFile(b) =>
                write!(f, "Block {} is not the start of a file", b),
            &Error::BadFormat(ref s) =>
                write!(f, "Bad card format: {}", s),
            &Error::BadLength =>
                write!(f, "Wrong card size (expected 128kB)"),
            &Error::IoError(ref e) => write!(f, "{}", e),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BlockUsage {
    Unused,
    First,
    Middle,
    Last,
}

pub struct BrokenSector {
    /// Absolute address of the broken sector.
    sector: u32,
    /// Absolute address of the replacement sector (in block 0).
    replacement: u32,
    /// Metadata sector checksum and whether it's valid or not.
    checksum: Result<u8, u8>,
}

impl BrokenSector {
    pub fn sector(&self) -> u32 {
        self.sector
    }

    pub fn replacement(&self) -> u32 {
        self.replacement
    }

    pub fn checksum(&self) -> Result<u8, u8> {
        self.checksum
    }
}

/// Size of a single sector.
pub const SECTOR_SIZE: usize = 128;

// There are 64 sectors per block
pub const SECTORS_PER_BLOCK: usize = 64;

/// Size of a single block in bytes. There are 16 blocks in a memory
/// card.
pub const BLOCK_SIZE: usize = SECTORS_PER_BLOCK * SECTOR_SIZE;

/// Total size of a memory card in bytes
pub const TOTAL_SIZE: usize = BLOCK_SIZE * 16;
