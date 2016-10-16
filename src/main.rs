use std::process::exit;
use std::path::Path;

extern crate encoding;

use mc::{MemoryCard, BLOCK_SIZE, SECTORS_PER_BLOCK};

mod mc;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();

    let cmd =
        match args.first() {
            Some(c) =>
                COMMANDS.iter()
                .find(|&&(s, _)| s == c)
                .map(|&(_, f)| f)
                .unwrap_or(help),
            None => help,
        };

    if let Err(_) = cmd(&args) {
        exit(1);
    }
}

fn help(_: &[String]) -> Result<(), ()> {
    println!("Usage: psx-mctool <command> [params [...]]");
    println!("");
    println!("Commands:");

    for &(c, _) in &COMMANDS {
        println!("  {}", c);
    }

    Err(())
}

fn expect_params(p: &[String],
                 expected: &[&'static str]) -> Result<(), ()> {
    if p.len() == expected.len() + 1 {
        Ok(())
    } else {
        println!("Usage: psx-mctool {}{}",
                 p[0],
                 expected.iter()
                 .fold(String::new(),
                       |f, s| format!("{} <{}>", f, s)));

        Err(())
    }
}

fn open_mc_file(path: &str) -> Result<MemoryCard, ()> {
    MemoryCard::from_path(Path::new(path))
        .map_err(|e| println!("Can't load {}: {}", path, e))
}

fn write_mc_file(mc: &MemoryCard, path: &str) -> Result<(), ()> {

    mc.dump_file(Path::new(path))
        .map_err(|e| println!("Can't write {}: {}", path, e))
}

fn list_blocks(p: &[String]) -> Result<(), ()> {
    try!(expect_params(p, &["memory-card"]));

    let mc = try!(open_mc_file(&p[1]));

    for block in 1..16 {
        let block = mc.block(block);

        let (free, usage) = block.state();

        println!("Block {}: {} ({:?})",
                 block.number(),
                 if free { "free" } else { "used" },
                 usage);

        println!("  File name: {}", block.file_name());

        let file_size = block.file_size() as usize;

        let file_blocks = file_size / BLOCK_SIZE;

        println!("  File size: {}B ({} block{})",
                 file_size,
                 file_blocks,
                 if file_blocks == 1 { "" } else { "s"});

        match block.next_block_raw() {
            Ok(n) => println!("  Next block: {}", n),
            // Regular "last block" value
            Err(0xffff) => (),
            Err(v) => println!("  Next block: 0x{:04x} (bogus!)",
                               v)
        }

        match block.checksum() {
            Ok(c) => println!("  Metadata checksum: 0x{:02x} (OK)", c),
            Err(c) => println!("  Metadata checksum: 0x{:02x} (INVALID)", c),
        }

        println!("");
    }

    Ok(())
}

fn list_broken_sectors(p: &[String]) -> Result<(), ()> {
    try!(expect_params(p, &["memory-card"]));

    let mc = try!(open_mc_file(&p[1]));

    let broken_sectors = mc.broken_sectors();

    if broken_sectors.is_empty() {
        println!("No broken sectors");
    } else {
        for s in &broken_sectors {
            let pos = s.sector() as usize;

            let block = pos / SECTORS_PER_BLOCK;
            let off = pos ^ SECTORS_PER_BLOCK;

            let block_str =
                if block > 15 {
                    format!("bogus!")
                } else {
                    format!("Block {} sector {}", block, off)
                };

            let csum_str =
                match s.checksum() {
                    Ok(c) => format!("Checksum OK (0x{:02x})", c),
                    Err(c) => format!("Bad checksum (0x{:02x})", c),
                };

            println!("Sector {} ({}) replaced by sector {} [{}]",
                     pos, block_str, s.replacement(), csum_str);
        }
    }

    Ok(())
}

fn do_list_files(p: &[String], show_deleted: bool) -> Result<(), ()> {
    try!(expect_params(p, &["memory-card"]));

    let mc = try!(open_mc_file(&p[1]));

    for &f in mc.files(show_deleted).iter() {
        list_file(&mc, f);
    }

    Ok(())
}

fn list_file(mc: &MemoryCard, f: u8) {
    let first_block = mc.block(f);

    if first_block.is_free() {
        println!("File #{} [DELETED]:", f);
    } else {
        println!("File #{}:", f);
    }

    println!("  File name: {}", first_block.file_name());

    match mc.file_title(f) {
        Ok(t) => println!("  Title: {}", t),
        Err(e) => println!("  Invalid title: {}", e),
    }

    let file_size = first_block.file_size() as usize;

    let file_blocks = file_size / BLOCK_SIZE;

    println!("  File size (reported): {}B ({} blocks)",
             file_size, file_blocks);

    match mc.file_blocks(f) {
        Ok(b) => {
            println!("  Blocks:{} ({} total)",
                     b.iter().fold(String::new(),
                                   |s, b| format!("{} {}", s, b)),
                     b.len());

            if b.len() != file_blocks {
                println!("  Reported file size does not match!");
            }
        }
        Err(e) => println!("  Block chain error: {}", e),
    }
}

fn list_files(p: &[String]) -> Result<(), ()> {
    do_list_files(p, false)
}

fn list_all_files(p: &[String]) -> Result<(), ()> {
    do_list_files(p, true)
}

fn format(p: &[String]) -> Result<(), ()> {
    try!(expect_params(p, &["memory-card"]));

    let mc = MemoryCard::new();

    write_mc_file(&mc, &p[1])
}

fn load_raw_file(p: &[String]) -> Result<(), ()> {
    try!(expect_params(p, &["memory-card", "raw-file", "file-name"]));

    let mut mc = try!(open_mc_file(&p[1]));

    let raw_file = Path::new(&p[2]);
    let file_name = p[3].as_bytes();

    let blocks =
        try!(mc.load_raw_file(raw_file, file_name)
             .map_err(|e| println!("Failed to load {}: {}",
                                   &p[2], e)));

    try!(write_mc_file(&mc, &p[1]));

    list_file(&mc, blocks[0]);

    Ok(())
}

static COMMANDS: [(&'static str, fn(&[String]) -> Result<(), ()>); 7] = [
    ("help", help),
    ("format", format),
    ("list-files", list_files),
    ("list-all-files", list_all_files),
    ("list-blocks", list_blocks),
    ("list-broken-sectors", list_broken_sectors),
    ("load-raw-file", load_raw_file),
];
